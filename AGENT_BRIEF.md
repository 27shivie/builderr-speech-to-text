# Builderr Challenge Draft - Local Hinglish Dictation Engine

## Title

Build the best local dictation engine for Indian-English and Hinglish work speech.

## In One Line

Build a local speech-to-text engine that turns English, Indian-English, and Hinglish work dictation into usable text faster and more accurately than generic local dictation tools, with no cloud calls during scoring.

## The Problem

People want Wispr Flow-style dictation for work: press a hotkey, speak naturally, and get clean text into Claude, Codex, Cursor, Slack, Docs, or email.

But in many companies, cloud dictation is not approved because the audio/text can contain company data.

So the real user need is:

```text
I need fast, accurate dictation that runs locally on my work laptop,
especially when I speak Indian-English or mix Hindi and English.
```

This is not a generic "wrap Whisper" problem. Generic local dictation already exists. The hard part is making local dictation good enough for messy builder speech:

- Indian accents.
- Hindi-English code-switching.
- Work terms like Codex, Cursor, Jira, PRD, API, rollback, latency, p95.
- Negations, numbers, dates, names, and acronyms.
- Low latency, because slow dictation is not habit-forming.
- No cloud calls, because the core use case is company-data safety.

## Why This Is A Real Challenge

The obvious alternatives already exist:

- Apple Dictation is built into every Mac.
- Handy is open-source, offline, and simple.
- OpenWhispr has local Whisper/Parakeet, hotkey dictation, notes, dictionary, and offline mode.
- TypeWhisper has local engines, Parakeet, WhisperKit, Apple Speech, history, dictionary, APIs, and model management.
- VoiceInk has native Mac UX, local models, modes, shortcuts, and dictionary features.
- Wispr Flow is the aspirational cloud UX benchmark.

So the challenge is not "can you run ASR locally?"

The challenge is:

```text
Can you beat the best local/free options on the exact speech pattern they do not optimize for:
Indian-English + Hinglish work dictation, under strict local-only and latency constraints?
```

This is a good Builderr challenge because it is objectively scorable:

- Same WAVs for every entrant.
- Hidden test clips.
- Clear output contract.
- Network blocked during official scoring.
- Latency measured by the harness.
- Score caps for critical meaning failures.
- Public leaderboard by accuracy, latency, reliability, and local-only proof.

## What To Build

Build the engine, not the whole app.

The product UX can be built later around the winning engine. The challenge only asks for a local transcription package with a clean command/API contract.

Entrants can use any local model or local model ensemble:

- whisper.cpp
- faster-whisper
- WhisperKit
- Parakeet
- Qwen3-ASR / Srota
- MLX models
- VAD/endpointing
- routing/selection/merging
- local cleanup models
- explicit user dictionary/profile terms

Cloud APIs are allowed during development if the builder wants, but the official scored run blocks outbound network after model warmup. Any network dependency during scoring fails.

## Submission Contract

Entrants provide a repo that exposes this command:

```bash
python -m solution.transcribe \
  --input /path/to/audio.wav \
  --mode auto \
  --output /path/to/result.json
```

Modes:

| Mode | Meaning |
| --- | --- |
| `auto` | Product default. Return the most useful work text. English/Roman output is preferred. |
| `fast` | Lowest-latency usable text. Must be strong for English and Indian-English. |
| `hinglish` | Higher-quality Hindi-English/code-switch mode. Can be slower, but must be better. |
| `verbatim` | Optional transcript-preserving mode. Devanagari is allowed here, not required for default. |

Required `result.json`:

```json
{
  "text": "Tell Cursor to update the PRD but do not change the deadline.",
  "mode_used": "auto",
  "language_guess": "hinglish",
  "timings_ms": {
    "total": 820,
    "asr": 690,
    "postprocess": 80
  },
  "raw_candidates": [
    {
      "engine": "whisper_cpp_small_translate",
      "text": "Tell Cursor to update the PRD but do not change the deadline."
    }
  ],
  "model_ids": ["whisper.cpp-small-q5", "srota-qwen3-hinglish"],
  "local_only": true
}
```

Optional but useful:

- confidence
- route chosen
- fallback reason
- detected repetition/degeneration
- dictionary terms matched/missed
- critical-fact warnings

## Current Baseline To Beat

RambleFix current engine:

```text
fast/default:
  resident whisper.cpp server, small model, auto language, translate mode

hinglish quality:
  Srota Qwen3 Hinglish via mlx-qwen3-asr
  + repetition guard
  + fast fallback
  + last-resort non-translate whisper.cpp fallback
```

Current measured performance:

| Slice | RambleFix current | What it means |
| --- | --- | --- |
| Public 120-row launch pool | useful `0.840`, p50 `0.705s`, p95 `3.696s`, hang `0.008` | Good overall, but p95 includes slower Hinglish quality path. |
| Fast local server on same pool | useful `0.766`, p50 `0.506s`, p95 `0.742s`, hang `0.000` | Strong latency baseline. |
| OpenSLR Hinglish 50-row slice | useful `0.849`, WER `0.261`, meaning `0.844`, terms `0.941`, p50 `2.523s`, p95 `4.109s`, blanks `0/50` | Current strongest wedge. |
| FLEURS English vs OpenWhispr small | RambleFix WER `0.082`, meaning `0.921`, p95 `0.550s`; OpenWhispr small WER `0.079`, meaning `0.929`, p95 `0.874s` | English is tied, not won. |
| YouTube English accents vs OpenWhispr small | RambleFix WER `0.272`, meaning `0.751`, p95 `0.710s`; OpenWhispr small WER `0.322`, meaning `0.705`, p95 `1.104s` | Promising, but partly driven by one competitor failure. |
| OpenSLR Hinglish vs OpenWhispr small | RambleFix WER `0.261`, meaning `0.844`, terms `0.941`, p95 `4.109s`; OpenWhispr small WER `1.002`, meaning `0.246`, terms `0.061`, p95 `8.163s` | Hinglish/code-switch engine wedge is real on this slice. |

Honest current read:

- Hinglish/code-switch is the real wedge.
- English is not solved as a superiority claim. It is roughly tied with local Whisper-small but faster.
- Parakeet and full TypeWhisper/OpenWhispr app paths remain serious unresolved baselines.
- The app UX is not part of this challenge. This is an engine challenge.

## Scoring

Each submission is scored on a hidden same-WAV set.

The hidden set should include:

- English office dictation.
- Indian-English office dictation.
- Hinglish/code-switch work speech.
- Fast rambles.
- Noisy laptop microphone clips.
- Dates, amounts, numbers, acronyms, product names, and proper names.
- Negation traps: "do not", "don't", "not June 21, June 24", "rollback is not approved".
- Cleanup traps: user says "make this cleaner" but the engine must preserve the spoken instruction accurately.

Score out of 100:

| Area | Weight | What is measured |
| --- | ---: | --- |
| Meaning accuracy | 40 | Does the output preserve what the user meant? WER is diagnostic, not the final score. |
| Critical facts and terms | 25 | Names, dates, numbers, negation, acronyms, work terms. Critical flips cap the row. |
| Latency and reliability | 20 | p50/p95 total runtime, blank rate, hang rate, timeout rate. |
| Local-only proof | 10 | Runs with network blocked after model warmup; no cloud dependency. |
| Auditability | 5 | Emits raw candidates, route, timings, model IDs, and fallback reason. |

Hard caps:

- Any date/number/negation/entity flip caps that row at `50%`.
- Blank output caps that row at `20%`.
- Repetition loop caps that row at `30%`.
- Output unrelated to the audio caps that row at `20%`.
- Network call during scored run is a fail.
- Exact hidden phrase rewrites, hidden answer maps, or hardcoded eval strings are a fail.

## Win Conditions

Minimum to be leaderboard-valid:

- Runs fully local with outbound network blocked after warmup.
- p95 total runtime under `5s` on the scored set.
- Hang/error rate below `1%`.
- No hardcoded phrase fixes or hidden answer maps.
- Emits the required JSON contract.

Strong submission target:

| Gate | Target |
| --- | --- |
| Overall useful score | `>=0.88` on hidden mixed set. |
| English / Indian-English | Within `0.02` useful score of the best local English engine, p95 fast mode `<800ms` warm. |
| Hinglish/code-switch | Meaning coverage `>=0.88`, term coverage `>=0.93`, and at least `0.05` useful-score lift over current RambleFix baseline. |
| Hinglish latency | p95 `<3.5s`, or fast draft under `1s` plus async final under `3.5s`. |
| Reliability | `0` blanks on launch smoke; hang/error rate `<1%`. |
| Offline | Full official eval passes with network blocked. |

Winner:

```text
Highest score after passing all gates, with ties broken by lower p95 latency.
```

## What Exactly Needs To Be Cracked

The core unsolved thing is engine routing and finalization:

```text
Give me sub-second local text for normal English/Indian-English,
but detect when the clip is Hinglish/code-switch or semantically risky,
then invoke a stronger local path that fixes meaning without hanging or hallucinating.
```

A winning approach probably needs:

- A fast foreground recognizer.
- A better Hinglish/code-switch recognizer or verifier.
- A route decision that does not run the slow model on every clip.
- A scorer that rejects repeat loops, blanks, and over-translation.
- Critical term/fact checks.
- Explicit dictionary/profile support without hidden phrase hacking.
- Offline model packaging/warmup.

## What Not To Build

Do not spend time on:

- A polished Mac UI.
- Meeting summaries.
- Cloud cleanup.
- A generic Whisper wrapper.
- Devanagari-first output for the default mode.
- Phrase hacks for the visible test set.

The winning engine can later be wrapped in a native Mac app with hotkey, paste, history, and mode switching.

## Starter Kit Shape

A good Builderr starter repo should include:

- `AGENT_BRIEF.md`: this challenge brief for AI assistants.
- `solution/transcribe.py`: required CLI contract.
- `baseline/ramblefix_current.py`: current RambleFix baseline wrapper.
- `preview.py`: public dev-set local preview.
- `scorecard.py`: useful-score, critical-fact, latency, and offline checks.
- `data/public_dev_manifest.json`: public clips and references.
- `tests/test_contract.py`: verifies JSON schema and local-only flags.
- `tests/test_no_network.py`: fails on outbound network during scoring.

## Builder-Facing Prompt

```text
Help me build a local dictation engine for this challenge.

Goal:
Given a WAV file of English, Indian-English, or Hinglish work speech, return usable work text locally.

Optimize for:
- preserving meaning
- preserving critical facts like dates, numbers, names, negation, acronyms, and work terms
- low p95 latency
- no cloud calls during scoring
- no hardcoded phrase fixes

Interface:
Implement `python -m solution.transcribe --input clip.wav --mode auto --output result.json`.

Current baseline:
RambleFix fast mode is sub-second but weaker on Hinglish.
RambleFix Hinglish mode is much better on code-switching but p95 is about 4.1s.

Find a better local engine/router/finalizer that beats this baseline.
```

## Self-Critique

This is a good Builderr challenge if we keep it narrow.

Good:

- Objective scoring is possible.
- Same-WAV hidden tests make the leaderboard fair.
- The problem is real for builders using AI at work.
- The winning artifact can directly become the product engine.
- The challenge avoids vague UX taste debates.

Risks:

- If the hidden corpus is too small, people overfit.
- If references are low quality, the leaderboard becomes noisy.
- If the scoring overweights WER, it will punish useful meaning-first outputs.
- If latency is measured only backend-side, it may not predict product feel.
- If we do not block network, the "local" claim becomes fake.

Mitigations:

- Use public dev data plus hidden human-gold clips.
- Score by meaning and critical facts, not WER alone.
- Publish latency p50/p95 and timeout rate.
- Run the official eval with network blocked.
- Keep UX out of scope except for engine latency.

