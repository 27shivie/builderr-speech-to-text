"""Regression tests for final-only dictation scoring."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming_scorecard import (  # noqa: E402
    CAP_SLOW_FINAL,
    CAP_VERY_SLOW_FINAL,
    end_to_final_points,
    rescore_saved_result,
    score_stream_clip,
)

GOLD = "rollback abhi mat karo pehle p95 check karlo"


def good_run(e2f_ms, final_text, *, partials=None):
    t_end = 10.0
    return {
        "t_start": 0.0,
        "t_end_audio": t_end,
        "partials": partials or [],
        "final": (t_end + e2f_ms / 1000.0, final_text),
        "dropped": False,
    }


def clip(runs, gold=GOLD, must=None):
    return score_stream_clip({"clip_id": "c", "gold": gold, "must_have": must or [], "runs": runs})


def test_latency_curve_uses_end_to_final_only():
    assert end_to_final_points(800) == 30.0
    assert 24.0 < end_to_final_points(1500) < 30.0
    assert end_to_final_points(2000) == 24.0
    assert end_to_final_points(3500) == 12.0
    assert abs(end_to_final_points(5000) - 3.6) < 1e-9
    assert end_to_final_points(6000) == 0.0


def test_partials_never_change_score_or_caps():
    no_partials = [good_run(800, GOLD) for _ in range(5)]
    thrashing_partials = [
        good_run(
            800,
            GOLD,
            partials=[
                (0.1, "completely wrong words", 22),
                (0.2, "rewritten again", 15),
                (9.9, "still wrong", 11),
            ],
        )
        for _ in range(5)
    ]
    clean = clip(no_partials)
    noisy = clip(thrashing_partials)
    assert clean.score == noisy.score
    assert clean.capped_at == noisy.capped_at
    assert clean.components == noisy.components
    assert clean.score == 100.0


def test_saved_partial_only_cap_is_removed_without_rerun():
    saved = {
        "clips": [{
            "clip_id": "stored",
            "meaning": 1.0,
            "wer": 0.0,
            "median_end_to_final_ms": 800,
            "final_text": GOLD,
            "capped_at": 70.0,
            "reasons": ["no useful committed partial", "no-useful-partial cap"],
            "components": {"facts": 20.0, "ttfs": 0.0, "churn": 0.0},
        }]
    }
    result = rescore_saved_result(saved)
    assert result["overall_score"] == 100.0
    assert result["clips"][0]["capped_at"] is None
    assert result["rescored_from_saved_artifact"] is True


def test_final_quality_and_latency_are_scored():
    fast = clip([good_run(800, GOLD) for _ in range(5)])
    slow = clip([good_run(4500, GOLD) for _ in range(5)])
    inaccurate = clip([good_run(800, "weather is sunny today") for _ in range(5)])
    assert fast.score > slow.score
    assert fast.score > inaccurate.score
    assert slow.capped_at == CAP_SLOW_FINAL


def test_missing_final_or_hang_scores_zero():
    blank = clip([good_run(800, "") for _ in range(5)])
    assert blank.score == 0.0

    runs = [good_run(800, GOLD) for _ in range(5)]
    runs[2]["dropped"] = True
    dropped = clip(runs)
    assert dropped.score == 0.0


def test_very_slow_final_is_capped():
    result = clip([good_run(6500, GOLD) for _ in range(5)])
    assert result.capped_at == CAP_VERY_SLOW_FINAL


def test_hindi_and_declared_romanized_reference():
    hindi = "यह एक सही हिंदी प्रतिलेख है"
    roman = "yeh ek sahi hindi pratilekh hai"
    hindi_result = score_stream_clip({
        "clip_id": "hi",
        "gold": hindi,
        "must_have": [],
        "runs": [good_run(800, hindi) for _ in range(5)],
    })
    assert hindi_result.capped_at is None
    assert hindi_result.meaning == 1.0

    roman_result = score_stream_clip({
        "clip_id": "hi-roman",
        "gold": hindi,
        "gold_alternatives": [roman],
        "must_have": [],
        "runs": [good_run(800, roman) for _ in range(5)],
    })
    assert roman_result.capped_at is None
    assert roman_result.meaning == 1.0
    assert roman_result.final_text == roman
    assert "reference" not in roman_result.components
