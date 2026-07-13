"""builderr · STREAMING dictation track — real-time feeder + metric capture.

This is the harness builderr runs. It launches the entrant's sealed
`solution/stream_server.py` on a loopback port, then for each clip:

  1. opens one WebSocket per run,
  2. feeds the WAV at 1x real time as 20ms binary PCM frames (absolute-time paced,
     so jitter never accumulates),
  3. captures the final event on a monotonic RECEIVE clock,
  4. repeats 5 times with per-run anti-replay jitter (gain + tiny resample),
  5. scores via streaming_scorecard.score_stream_run — latency on the 5-run
     median, quality on the median-latency run's final.

After READY + a warm-up clip, it calls block_network() so the scored run is
offline (loopback to a local ASR server stays allowed — see offline_guard.py).

Adapted from RambleFix's run_streaming_latency_eval.py. No t_ms / seq / pcm_b64:
every scored timing is the evaluator's own monotonic receive time.

Usage (entrant-facing wrapper is preview_stream.py):
    python evaluator.py --manifest samples/manifest.json --runs 5
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from streaming_scorecard import score_stream_run  # noqa: E402
from offline_guard import block_network  # noqa: E402

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

FRAME_MS = 20
SR = 16000
FRAME_BYTES = int(SR * FRAME_MS / 1000) * 2  # 20ms s16le mono = 640 bytes
DEFAULT_WARMUP_WAV = os.path.join(HERE, "samples", "fleurs_en_us_test_1904.wav")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def _read_pcm_16k_mono(wav_path: str) -> bytes:
    with wave.open(wav_path, "rb") as r:
        sr = r.getframerate()
        ch = r.getnchannels()
        sw = r.getsampwidth()
        raw = r.readframes(r.getnframes())
    if np is None:
        return raw  # best effort; assume already 16k mono s16le
    a = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != SR:
        a = _resample(a, sr, SR)
    return a.astype(np.int16).tobytes()


def _resample(a, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return a
    n_out = int(round(len(a) * dst_sr / src_sr))
    if n_out <= 0:
        return a[:0]
    xp = np.linspace(0.0, 1.0, num=len(a), endpoint=False)
    x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x, xp, a.astype(np.float32)).astype(np.int16)


def _jitter(pcm: bytes, seed: int) -> bytes:
    """Per-run anti-replay perturbation: random gain +/-0.5 dB and +/-0.3%
    resample, deterministic per (clip, run). Defeats fingerprint memoization
    across the 5 warm serial runs without changing the words."""
    if np is None:
        return pcm
    rng = np.random.default_rng(seed)
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    gain_db = rng.uniform(-0.5, 0.5)
    a = a * (10.0 ** (gain_db / 20.0))
    speed = 1.0 + rng.uniform(-0.003, 0.003)
    if abs(speed - 1.0) > 1e-6:
        a16 = np.clip(a, -32768, 32767).astype(np.int16)
        a = _resample(a16, SR, int(round(SR / speed))).astype(np.float32)
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


def _jitter_seed(clip_id: str, attempt: int) -> int:
    """Stable across Python processes and machines for reproducible reruns."""
    payload = f"builderr-stt-v2\0{clip_id}\0{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _wire_clip_id(clip_id: str, secret: bytes) -> str:
    return hashlib.blake2s(clip_id.encode("utf-8"), key=secret, digest_size=12).hexdigest()


def _frames(pcm: bytes):
    for i in range(0, len(pcm), FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


# --------------------------------------------------------------------------
# one streamed run
# --------------------------------------------------------------------------

async def _run_once(uri: str, clip_id: str, pcm: bytes) -> dict:
    import websockets
    partials: list[tuple[float, str, int]] = []
    final = None
    dropped = False
    t_start = None
    t_end_audio = None
    pace_ok = True
    expected_s = len(pcm) / (SR * 2)

    try:
        async with websockets.connect(uri, max_size=None) as ws:
            await ws.send(json.dumps({
                "type": "start", "sample_rate": SR, "format": "pcm_s16le",
                "channels": 1, "clip_id": clip_id}))

            async def _receiver():
                nonlocal final
                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    now = time.monotonic()
                    if msg.get("type") == "partial":
                        partials.append((now, msg.get("text", ""), int(msg.get("stable_chars", 0))))
                    elif msg.get("type") == "final":
                        final = (now, msg.get("text", ""))
                        return

            recv_task = asyncio.create_task(_receiver())

            # absolute-time paced feeder — no drift accumulation
            t_start = time.monotonic()
            t_send = t_start
            dt = FRAME_MS / 1000.0
            for frame in _frames(pcm):
                await ws.send(frame)
                t_send += dt
                delay = t_send - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(json.dumps({"type": "end"}))
            t_end_audio = time.monotonic()

            # pace sanity: total send time within +/-5% of clip duration
            sent_s = t_end_audio - t_start
            pace_ok = (np is None) or (abs(sent_s - expected_s) <= 0.05 * max(0.001, expected_s))

            try:
                await asyncio.wait_for(recv_task, timeout=20.0)
            except asyncio.TimeoutError:
                dropped = True
                recv_task.cancel()
    except Exception:  # noqa: BLE001 - any transport failure => dropped clip
        dropped = True

    return {
        "clip_id": clip_id,
        "t_start": t_start,
        "t_end_audio": t_end_audio,
        "partials": partials,
        "final": final,
        "dropped": dropped or final is None,
        "pace_ok": pace_ok,
    }


async def _capture_clip(uri: str, clip: dict, runs: int, wire_secret: bytes) -> dict:
    pcm = _read_pcm_16k_mono(clip["_wav"])
    captured = []
    attempt_seed = 0
    wire_id = _wire_clip_id(clip["clip_id"], wire_secret)
    while len(captured) < runs and attempt_seed < runs * 4:
        seed = _jitter_seed(clip["clip_id"], attempt_seed)
        attempt_seed += 1
        run = await _run_once(uri, wire_id, _jitter(pcm, seed))
        # pace sanity: an out-of-band run is discarded and re-run, never scored
        if not run.get("pace_ok", True) and not run.get("dropped"):
            continue
        captured.append(run)
    return {
        # Never put source-dataset IDs in result artifacts. They can be enough to
        # recover public-corpus references even when the gold text itself is absent.
        "clip_id": wire_id,
        "gold": clip.get("gold", ""),
        "gold_alternatives": clip.get("gold_alternatives", []),
        "must_have": clip.get("must_have", []),
        "category": clip.get("category", ""),
        "runs": captured,
    }


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _sandbox_profile(deny_read_roots: list[str]) -> str:
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        '(allow network-outbound (remote ip "localhost:*"))',
        '(allow network-inbound (local ip "localhost:*"))',
    ]
    for root in sorted({str(Path(value).expanduser().resolve()) for value in deny_read_roots if value}):
        rules.append(f"(deny file-read* (subpath {json.dumps(root)}))")
    return "".join(rules)


def _drain_server_output(stream, log_path: str, initial_lines: list[str]) -> None:
    with open(log_path, "w", encoding="utf-8") as log:
        log.writelines(initial_lines)
        log.flush()
        for line in iter(stream.readline, ""):
            log.write(line)
            log.flush()


def _start_server(module: str, port: int, *, enforce_offline: bool,
                  deny_read_roots: list[str], server_log: str) -> subprocess.Popen:
    command = [sys.executable, "-m", module, "--host", "127.0.0.1", "--port", str(port)]
    if enforce_offline:
        if sys.platform != "darwin" or not os.path.exists(SANDBOX_EXEC):
            raise RuntimeError("official scoring requires macOS sandbox-exec process isolation")
        command = [SANDBOX_EXEC, "-p", _sandbox_profile(deny_read_roots), *command]

    env = os.environ.copy()
    if enforce_offline:
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        })
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HERE,
        env=env, text=True, bufsize=1)
    assert proc.stdout is not None

    # Wait without blocking forever when the child never emits a newline.
    deadline = time.monotonic() + 60
    initial_lines: list[str] = []
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "".join(initial_lines)[-4000:]
            raise RuntimeError(f"stream server exited before READY\n{tail}")
        ready, _, _ = select.select([proc.stdout], [], [], 0.25)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            continue
        initial_lines.append(line)
        if line.startswith(f"READY port={port}"):
            Path(server_log).parent.mkdir(parents=True, exist_ok=True)
            thread = threading.Thread(
                target=_drain_server_output,
                args=(proc.stdout, server_log, initial_lines),
                name="builderr-stt-server-log",
                daemon=True,
            )
            thread.start()
            return proc
    proc.terminate()
    raise RuntimeError("stream server did not print READY within 60 seconds")


def _resolve_audio(clip: dict, manifest_path: str) -> str:
    base = Path(manifest_path).expanduser().resolve().parent
    candidates: list[Path] = []
    for value in (clip.get("audio_local"), clip.get("audio")):
        if not value:
            continue
        path = Path(str(value)).expanduser()
        candidates.append(path if path.is_absolute() else base / path)
        candidates.append(Path(HERE) / path)
    candidates.extend([
        base / "audio" / f"{clip['clip_id']}.wav",
        base / f"{clip['clip_id']}.wav",
    ])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return str(resolved)
    searched = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(f"missing audio for {clip['clip_id']}; searched:\n{searched}")


def _select_manifest(manifest: list[dict], *, limit: int = 0,
                     per_category: int = 0) -> list[dict]:
    if limit and per_category:
        raise ValueError("use either --limit or --per-category, not both")
    if limit:
        return manifest[:limit]
    if not per_category:
        return manifest
    counts: dict[str, int] = {}
    selected = []
    for row in manifest:
        category = str(row.get("category") or "unknown")
        if counts.get(category, 0) >= per_category:
            continue
        selected.append(row)
        counts[category] = counts.get(category, 0) + 1
    return selected


async def _evaluate(manifest_path: str, server_module: str, runs: int,
                    enforce_offline: bool, warmup_wav: str,
                    deny_read_roots: list[str], server_log: str,
                    limit: int = 0, per_category: int = 0) -> dict:
    manifest = json.load(open(manifest_path))
    manifest = _select_manifest(manifest, limit=limit, per_category=per_category)
    for c in manifest:
        c["_wav"] = _resolve_audio(c, manifest_path)

    warmup_path = str(Path(warmup_wav).expanduser().resolve())
    if not os.path.isfile(warmup_path):
        raise FileNotFoundError(f"missing dedicated warmup WAV: {warmup_path}")

    port = _free_port()
    private_roots = [str(Path(manifest_path).expanduser().resolve().parent), *deny_read_roots]
    proc = _start_server(
        server_module, port, enforce_offline=enforce_offline,
        deny_read_roots=private_roots, server_log=server_log)
    uri = f"ws://127.0.0.1:{port}"
    try:
        # Warm on a dedicated public clip, never on hidden scored audio.
        warm = _read_pcm_16k_mono(warmup_path)
        await _run_once(uri, "__warmup__", warm)
        if enforce_offline:
            block_network()  # defense-in-depth for the evaluator process itself

        wire_secret = os.urandom(32)
        clips = [await _capture_clip(uri, c, runs, wire_secret) for c in manifest]
        result = score_stream_run(clips)
        result["harness"] = {
            "version": "builderr-stt-v2",
            "process_network_isolation": enforce_offline,
            "sealed_server": server_module == "solution.stream_server",
            "dedicated_warmup": warmup_path,
            "opaque_wire_ids": True,
            "stable_jitter": True,
            "server_log": str(Path(server_log).resolve()),
        }
        return result
    finally:
        if enforce_offline:
            from offline_guard import restore_network
            restore_network()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "samples/manifest.json"))
    ap.add_argument("--server-module", default="solution.stream_server")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="dev/smoke only; official run uses all rows")
    ap.add_argument("--per-category", type=int, default=0,
                    help="balanced dev/screening slice only; official run uses all rows")
    ap.add_argument("--no-offline", action="store_true",
                    help="skip block_network() (dev only; official run always enforces)")
    ap.add_argument("--warmup-wav", default=DEFAULT_WARMUP_WAV,
                    help="dedicated public, unscored warmup WAV; never use hidden audio")
    ap.add_argument("--deny-read-root", action="append", default=[],
                    help="additional path hidden from the entrant server; repeatable")
    ap.add_argument("--server-log", default=os.path.join(HERE, "server.log"))
    ap.add_argument("--output-json", help="private result artifact path")
    ap.add_argument("--json", action="store_true", help="dump full JSON result")
    args = ap.parse_args()

    res = asyncio.run(_evaluate(args.manifest, args.server_module, args.runs,
                                enforce_offline=not args.no_offline,
                                warmup_wav=args.warmup_wav,
                                deny_read_roots=args.deny_read_root,
                                server_log=args.server_log,
                                limit=args.limit,
                                per_category=args.per_category))
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(f"\n  streaming score   {res['overall_score']}/100")
    print(f"  meaning {res['meaning_mean']}   WER {res['wer_mean']}")
    print(f"  median end-to-final {res['median_end_to_final_ms']}ms")
    print(f"  final-ok {res['final_ok_rate']}   clips capped {res['clips_capped']}/{res['n']}")
    for c in res["clips"]:
        flag = f"  capped@{c['capped_at']}" if c["capped_at"] else ""
        print(f"    {c['clip_id'][:28]:28s} score {c['score']:6}  final {c['median_end_to_final_ms']}ms"
              f"{flag}  {';'.join(c['reasons'][:2])}")


if __name__ == "__main__":
    main()
