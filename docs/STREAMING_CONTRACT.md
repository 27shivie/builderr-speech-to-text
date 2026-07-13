# Live dictation contract

This is the single source of truth for the $500 dictation challenge. If another
file disagrees with it, this file wins.

There is one prize and one score out of 100. RambleFix is the benchmark to beat
and cannot win the prize.

## What you build

Implement `draft()` in [`solution/draft.py`](../solution/draft.py):

```python
def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    """Return your best transcript of all audio received so far."""
```

The sealed harness sends 16 kHz mono PCM audio in real time. It calls `draft()`
while audio arrives and once after the user stops. Your final call should return
the text that would be pasted into the user's active app.

Keep Hindi-English code-switching faithful. Write what was said; do not translate
the mix into English.

`stable_chars` and intermediate drafts are optional UI metadata. They receive no
points, trigger no caps, and have no effect on ranking.

## Wire protocol

Evaluator to solution:

```jsonc
{"type":"start","sample_rate":16000,"format":"pcm_s16le","channels":1,"clip_id":"<opaque>"}
// binary 20 ms PCM frames at 1x real time
{"type":"end"}
```

Solution to evaluator:

```jsonc
{"type":"partial","text":"rollback abhi mat","stable_chars":11}  // optional, unscored
{"type":"final","text":"rollback abhi mat karo, pehle p95 check karlo"}  // exactly one
{"type":"meta","model_ids":["..."],"local_only":true}  // optional, unscored audit
```

Entrant timestamps are ignored. The evaluator measures final latency on its own
monotonic receive clock.

## Scoring

| Metric | Weight | What counts |
|---|---:|---|
| Final meaning and fidelity | 50 | Does the final preserve what the speaker meant, including the language mix? |
| Critical facts and terms | 20 | Numbers, negation, names, and required terms must survive in the final. |
| Final paste latency | 30 | Median time from the last audio frame to receipt of the final over five runs. |

Only the final transcript and final paste latency count. There is no scoring for
time-to-first-partial, partial quality, partial stability, or revision churn.

Quality is judged on the final from the median-latency run. Final paste latency
uses the median over five repeated runs.

### Final latency points

```text
<= 1000 ms       30 points
1000-2000 ms     linear 30 to 24
2000-3500 ms     linear 24 to 12
3500-5000 ms     linear 12 to 3.6
> 5000 ms        0 points
```

The goal is a useful final roughly two seconds after the user stops. This is not
about producing text early while the user is still speaking.

### Caps and failures

| Final result | Effect |
|---|---:|
| Any timed run drops or hangs | that clip scores 0 |
| Blank final | that clip scores 0 |
| Repetition loop in final | cap 30 |
| Final unrelated to audio | cap 20 |
| Critical fact flip | cap 50 |
| Median final latency over 4 seconds | cap 70 |
| Median final latency over 6 seconds | cap 50 |
| Non-loopback network call | evaluation fails |

Intermediate drafts never affect these caps.

## Evaluation harness

For each clip, the official evaluator:

1. Feeds audio at 1x real time in 20 ms frames.
2. Measures the final received after `end` on its own monotonic clock.
3. Repeats five times with small deterministic gain and resampling changes to
   prevent audio fingerprint lookup.
4. Discards and repeats a run if the harness itself failed to feed audio at the
   correct pace. This validates the evaluator and is not an entrant metric.
5. Scores final quality on the median-latency run and latency on the five-run
   median.

The official run uses a pinned MacBook Pro 14-inch (2021), Apple M1 Pro, 32 GB
RAM, with acceleration available and outbound network blocked after warmup.

## RambleFix benchmark and corpus rule

RambleFix and every entrant must run through the same evaluator on the **same hidden corpus**
and frozen machine. The official manifest is
`data/hidden/manifest.json`: 96 rows covering English, Hindi, mixed Hindi-English,
and longer English speech. If that corpus or scoring contract changes, RambleFix
must be rescored from saved compatible captures or rerun before a prize decision.

RambleFix is useful but not the target ceiling: its faithful final has historically
been slower than the roughly two-second product goal. A winning submission needs
to preserve mixed-language meaning and produce its final faster.

Any older result measured on a smaller corpus is a **component check**, **not the final payout line**.

## Run locally

```bash
pip install -r requirements.txt -r requirements-streaming.txt
python preview_stream.py
pytest tests/test_streaming_scorecard.py
pytest tests/test_stream_contract.py
```
