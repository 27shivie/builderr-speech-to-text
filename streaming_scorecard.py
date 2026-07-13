"""builderr dictation track scoring: final transcript quality and paste latency.

Only the result available when the user stops speaking is scored. Intermediate
partials are optional protocol messages and never affect points, caps, or
qualification.

ONE $500 prize, ONE combined score out of 100:

    Final meaning / fidelity                         50
    Critical facts & terms (numbers / negation)     20
    End-to-final latency (release -> final paste)   30

Quality is judged on the final from the median-latency run. End-to-final latency
uses the median over the repeated anti-replay runs. A blank, hung, or dropped
final scores zero because the product produced nothing the user could paste.

Pure standard library. All timings come from the evaluator's monotonic receive
clock; entrant-supplied timestamps are ignored.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from scorecard import (
    critical_flip,
    has_repetition_loop,
    judge_meaning,
    normalize,
    phonetic_token_f1,
    phonetic_wer,
    wer,
)

W_MEANING = 50.0
W_FACTS = 20.0
W_END_TO_FINAL = 30.0
assert W_MEANING + W_FACTS + W_END_TO_FINAL == 100.0

CAP_SLOW_FINAL = 70.0
CAP_VERY_SLOW_FINAL = 50.0
CAP_FACT_FLIP = 50.0
CAP_LOOP = 30.0
CAP_UNRELATED = 20.0


def _references(gold: str | list[str]) -> list[str]:
    values = [gold] if isinstance(gold, str) else list(gold)
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value.strip()))


def _best_reference(gold: str | list[str], pred: str) -> tuple[str, int, float, float]:
    refs = _references(gold)
    if not refs:
        return "", 0, 0.0, 1.0
    candidates = []
    for index, reference in enumerate(refs):
        meaning = judge_meaning(reference, pred)
        error = wer(reference, pred)
        if index > 0:
            meaning = max(meaning, phonetic_token_f1(reference, pred))
            error = min(error, phonetic_wer(reference, pred))
        candidates.append((reference, index, meaning, error))
    return max(candidates, key=lambda item: (item[2], -item[3]))


def end_to_final_ms(run) -> float | None:
    """Milliseconds from final audio frame sent to final text received."""
    final = run.get("final")
    if not final or run.get("dropped"):
        return None
    recv, _text = final
    t_end = run.get("t_end_audio")
    if t_end is None:
        return None
    return max(0.0, (recv - t_end) * 1000.0)


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y1
    frac = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * max(0.0, min(1.0, frac))


def end_to_final_points(median_ms: float | None) -> float:
    """Score only how long the user waits after releasing the dictation key."""
    if median_ms is None:
        return 0.0
    if median_ms <= 1000:
        return W_END_TO_FINAL
    if median_ms <= 2000:
        return _lerp(median_ms, 1000, 2000, 30.0, 24.0)
    if median_ms <= 3500:
        return _lerp(median_ms, 2000, 3500, 24.0, 12.0)
    if median_ms <= 5000:
        return _lerp(median_ms, 3500, 5000, 12.0, 3.6)
    return 0.0


@dataclass
class StreamClipResult:
    clip_id: str
    score: float
    capped_at: float | None
    reasons: list[str] = field(default_factory=list)
    meaning: float = 0.0
    wer: float = 0.0
    median_end_to_final_ms: float | None = None
    final_ok: bool = True
    final_text: str = ""
    reference_variant: int = 0
    components: dict = field(default_factory=dict)


def _median(vals: list[float]) -> float | None:
    return None if not vals else float(statistics.median(vals))


def _median_latency_run(runs):
    """Return the run at the measured median end-to-final latency."""
    measurable = [(end_to_final_ms(run), run) for run in runs]
    measurable = [(value, run) for value, run in measurable if value is not None]
    if not measurable:
        for run in runs:
            if run.get("final") and not run.get("dropped"):
                return run
        return runs[0] if runs else None
    measurable.sort(key=lambda item: item[0])
    return measurable[(len(measurable) - 1) // 2][1]


def score_stream_clip(clip) -> StreamClipResult:
    """Score one clip from final text and end-to-final latency only.

    `partials` may be present on runs for protocol compatibility. Their timing,
    text, and stability metadata are deliberately never read here.
    """
    clip_id = clip.get("clip_id", "")
    gold = clip.get("gold", "")
    refs = [gold, *(clip.get("gold_alternatives") or [])]
    must_have = clip.get("must_have") or []
    runs = clip.get("runs") or []
    reasons: list[str] = []

    latencies = [value for value in (end_to_final_ms(run) for run in runs) if value is not None]
    median_latency = _median(latencies)
    median_run = _median_latency_run(runs)
    final_text = ""
    if median_run is not None and median_run.get("final") and not median_run.get("dropped"):
        final_text = median_run["final"][1] or ""

    score_reference, reference_variant, meaning, error = _best_reference(refs, final_text)
    flipped, fact_reasons = critical_flip(score_reference, final_text, must_have)
    reasons += fact_reasons

    meaning_points = W_MEANING * meaning
    fact_points = 0.0 if flipped else W_FACTS
    latency_points = end_to_final_points(median_latency)
    base = meaning_points + fact_points + latency_points

    any_dropped = any(run.get("dropped") for run in runs)
    blank_final = not normalize(final_text)
    loop = has_repetition_loop(final_text)
    final_ok = not (any_dropped or blank_final or loop)

    cap = None

    def apply_cap(value: float, reason: str):
        nonlocal cap
        cap = value if cap is None else min(cap, value)
        reasons.append(reason)

    if any_dropped:
        apply_cap(0.0, "clip=0: final dropped or timed out")
    if blank_final:
        apply_cap(0.0, "clip=0: blank final")
    elif loop:
        apply_cap(CAP_LOOP, "repetition loop in final")
    elif error > 0.9:
        apply_cap(CAP_UNRELATED, f"final unrelated to audio (WER {error:.2f})")
    if flipped:
        apply_cap(CAP_FACT_FLIP, "critical fact flip on final")
    if median_latency is not None and median_latency > 6000:
        apply_cap(CAP_VERY_SLOW_FINAL, f"median final latency {median_latency:.0f}ms > 6000ms")
    elif median_latency is not None and median_latency > 4000:
        apply_cap(CAP_SLOW_FINAL, f"median final latency {median_latency:.0f}ms > 4000ms")

    score = min(base, cap) if cap is not None else base
    return StreamClipResult(
        clip_id=clip_id,
        score=round(score, 2),
        capped_at=cap,
        reasons=reasons,
        meaning=round(meaning, 3),
        wer=round(error, 3),
        median_end_to_final_ms=None if median_latency is None else round(median_latency, 1),
        final_ok=final_ok,
        final_text=final_text,
        reference_variant=reference_variant,
        components={
            "meaning": round(meaning_points, 2),
            "facts": round(fact_points, 2),
            "end_to_final": round(latency_points, 2),
        },
    )


def score_stream_run(clips) -> dict:
    results = [score_stream_clip(clip) for clip in clips]
    count = len(results) or 1

    def average(getter):
        values = [getter(result) for result in results if getter(result) is not None]
        return round(sum(values) / len(values), 1) if values else None

    return {
        "overall_score": round(sum(result.score for result in results) / count, 2),
        "meaning_mean": round(sum(result.meaning for result in results) / count, 3),
        "wer_mean": round(sum(result.wer for result in results) / count, 3),
        "median_end_to_final_ms": average(lambda result: result.median_end_to_final_ms),
        "final_ok_rate": round(sum(1 for result in results if result.final_ok) / count, 3),
        "clips_capped": sum(1 for result in results if result.capped_at is not None),
        "n": count,
        "clips": [result.__dict__ for result in results],
    }


def rescore_saved_result(saved: dict) -> dict:
    """Apply the final-only formula to an existing result artifact.

    This is for contract migrations where rerunning an engine is unnecessary.
    It uses only diagnostics already captured from the final: meaning, facts,
    WER, final text, drop/hang reasons, and end-to-final latency.
    """
    rescored = []
    for old in saved.get("clips", []):
        old_components = old.get("components") or {}
        old_reasons = [str(reason) for reason in old.get("reasons", [])]
        reason_text = " ".join(old_reasons).lower()
        final_text = old.get("final_text") or ""
        meaning = float(old.get("meaning") or 0.0)
        error = float(old.get("wer") or 0.0)
        latency = old.get("median_end_to_final_ms")
        latency = None if latency is None else float(latency)
        fact_flip = float(old_components.get("facts", W_FACTS)) <= 0.0
        dropped = "connection drop/hang" in reason_text or "final dropped or timed out" in reason_text
        blank = not normalize(final_text)
        loop = has_repetition_loop(final_text)

        reasons = []
        cap = None

        def apply_cap(value: float, reason: str):
            nonlocal cap
            cap = value if cap is None else min(cap, value)
            reasons.append(reason)

        if dropped:
            apply_cap(0.0, "clip=0: final dropped or timed out")
        if blank:
            apply_cap(0.0, "clip=0: blank final")
        elif loop:
            apply_cap(CAP_LOOP, "repetition loop in final")
        elif error > 0.9:
            apply_cap(CAP_UNRELATED, f"final unrelated to audio (WER {error:.2f})")
        if fact_flip:
            apply_cap(CAP_FACT_FLIP, "critical fact flip on final")
        if latency is not None and latency > 6000:
            apply_cap(CAP_VERY_SLOW_FINAL, f"median final latency {latency:.0f}ms > 6000ms")
        elif latency is not None and latency > 4000:
            apply_cap(CAP_SLOW_FINAL, f"median final latency {latency:.0f}ms > 4000ms")

        components = {
            "meaning": round(W_MEANING * meaning, 2),
            "facts": 0.0 if fact_flip else W_FACTS,
            "end_to_final": round(end_to_final_points(latency), 2),
        }
        base = sum(components.values())
        score = min(base, cap) if cap is not None else base
        rescored.append({
            "clip_id": old.get("clip_id", ""),
            "score": round(score, 2),
            "capped_at": cap,
            "reasons": reasons,
            "meaning": round(meaning, 3),
            "wer": round(error, 3),
            "median_end_to_final_ms": None if latency is None else round(latency, 1),
            "final_ok": not (dropped or blank or loop),
            "final_text": final_text,
            "reference_variant": old.get("reference_variant", 0),
            "components": components,
        })

    count = len(rescored) or 1
    latencies = [clip["median_end_to_final_ms"] for clip in rescored if clip["median_end_to_final_ms"] is not None]
    return {
        "scoring_contract": "final-only-v1",
        "rescored_from_saved_artifact": True,
        "overall_score": round(sum(clip["score"] for clip in rescored) / count, 2),
        "meaning_mean": round(sum(clip["meaning"] for clip in rescored) / count, 3),
        "wer_mean": round(sum(clip["wer"] for clip in rescored) / count, 3),
        "median_end_to_final_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "final_ok_rate": round(sum(1 for clip in rescored if clip["final_ok"]) / count, 3),
        "clips_capped": sum(1 for clip in rescored if clip["capped_at"] is not None),
        "n": count,
        "clips": rescored,
    }


if __name__ == "__main__":
    import json
    import sys

    clips = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else []
    print(json.dumps(score_stream_run(clips), indent=2))
