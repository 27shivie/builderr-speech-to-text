"""Runtime-agnostic ASR backend interface.

ONE seam so the decode runtime (faster-whisper today, MLX later) can be swapped
without touching draft(), the harness, or the benchmark. A backend is anything
that can load a Whisper-family model and transcribe a PCM buffer to text.

Selection is by env var so bench/models.py can A/B runtimes through the identical
harness:

    BUILDERR_BACKEND = faster-whisper (default) | mlx
    BUILDERR_MODEL   = small (default) | medium | large-v3-turbo | ...
    BUILDERR_COMPUTE = int8 (default, faster-whisper only)
    BUILDERR_DEVICE  = cpu  (default, faster-whisper only)

CONTRACT every backend obeys:
  - .name           -> str, stable id for reporting
  - .load()         -> None, does ALL cold-start work (import, weights, compile,
                       and a throwaway inference to warm kernels). Called once,
                       during the harness's unscored warmup clip. After load()
                       returns, transcribe() must be warm.
  - .transcribe(pcm)-> str, warm inference on int16-mono-16k PCM bytes.

Only warm .transcribe() is scored (the evaluator is a single long-lived process;
cold start is absorbed by the warmup clip). Backends optimize warm inference;
cold cost is paid once and does not count.
"""
from __future__ import annotations

import os
from typing import Protocol


class Backend(Protocol):
    name: str
    def load(self) -> None: ...
    def transcribe(self, pcm: bytes) -> str: ...


# --------------------------------------------------------------------------
# faster-whisper (CTranslate2) — the only implemented backend today.
# CPU-only on macOS (no Metal/ANE). Portable Mac+Linux, MIT, zero build step.
# --------------------------------------------------------------------------
class FasterWhisperBackend:
    def __init__(self, model: str, compute: str = "int8", device: str = "cpu") -> None:
        self.name = f"faster-whisper/{model}/{compute}/{device}"
        self._model_id = model
        self._compute = compute
        self._device = device
        self._model = None
        self._np = None

    def load(self) -> None:
        import numpy as np
        from faster_whisper import WhisperModel
        self._np = np
        self._model = WhisperModel(
            self._model_id, device=self._device, compute_type=self._compute)
        # warm the graph with a throwaway decode so first SCORED call is warm.
        silence = self._np.zeros(16000, dtype=self._np.float32)
        list(self._model.transcribe(silence, language="en", task="transcribe")[0])

    def transcribe(self, pcm: bytes) -> str:
        audio = self._np.frombuffer(pcm, dtype=self._np.int16).astype(self._np.float32) / 32768.0
        if audio.size == 0:
            return ""
        lang = os.environ.get("BUILDERR_LANG") or None
        segments, _info = self._model.transcribe(audio, language=lang, task="transcribe")
        return " ".join(s.text for s in segments).strip()


# --------------------------------------------------------------------------
# MLX Whisper — STUB. Not implemented. Do NOT fill this in until the
# faster-whisper accuracy-latency frontier proves CPU inference blocks a
# meaning-superior model inside the 4000ms budget.
#
# When that day comes, the ENTIRE integration is the three methods below:
#   load():        import mlx_whisper; prefetch the mlx-community/<model> repo;
#                  run one throwaway transcribe to compile Metal kernels.
#                  (Whisper pads to 30s so input shape is constant -> kernels
#                  compile once. Verify no per-clip recompile.)
#   transcribe():  mlx_whisper.transcribe(audio, path_or_hf_repo=<repo>,
#                  language=..., word_timestamps=False)  # timestamps are a
#                  known MLX perf trap (27s vs 3s) -- keep them OFF.
#   name:          f"mlx/{model}"
# Nothing else in the codebase changes. That is the point of this file.
# --------------------------------------------------------------------------
class MLXWhisperBackend:
    def __init__(self, model: str) -> None:
        self.name = f"mlx/{model}"
        self._model_id = model

    def load(self) -> None:
        raise NotImplementedError(
            "MLX backend is a deliberate stub. Integrate only if the "
            "faster-whisper frontier shows CPU blocks a meaning-superior model "
            "within 4000ms. See solution/backends.py for the 3-method recipe.")

    def transcribe(self, pcm: bytes) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Oriserve Whisper-Hindi2Hinglish — transformers/PyTorch, device=mps on Mac.
# This is the FIDELITY candidate: purpose-built for code-switched Hindi-English.
# License Apache-2.0 (ship-safe). It ROMANIZES output ("Is tutorial mein ..."),
# so under the current scorer it is graded against the romanized gold_alternative
# via phonetic matching -- NOT the Devanagari reference. Two consequences to
# watch in the A/B (read the per-clip CAP column, not just mean score):
#   - English must_have terms are EXACT substring, unaffected by phonetics -> a
#     romanized decode can still trip a fact-flip cap (50).
#   - This is a different runtime from faster-whisper; only warm inference is
#     scored (single long-lived server), so the heavy transformers import/load
#     is paid once on the warmup clip. load() below forces that warm-up.
# BUILDERR_MODEL selects the variant: Swift (fastest) | Prime | Apex (best).
# --------------------------------------------------------------------------
class OriserveBackend:
    _REPO = {
        "swift": "Oriserve/Whisper-Hindi2Hinglish-Swift",
        "prime": "Oriserve/Whisper-Hindi2Hinglish-Prime",
        "apex":  "Oriserve/Whisper-Hindi2Hinglish-Apex",
    }

    def __init__(self, variant: str = "swift") -> None:
        v = variant.lower()
        if v not in self._REPO:
            v = "swift"
        self.name = f"oriserve/{v}"
        self._repo = self._REPO[v]
        self._pipe = None
        self._np = None

    def load(self) -> None:
        import os as _os
        import numpy as np
        import torch
        # FIX (Jul 16, confirmed by log): AutoModel/AutoProcessor.from_pretrained()
        # does a HEAD request to check for updates even when weights are already
        # HF-cached locally. Under the offline sandbox this DNS call fails after
        # 5 retries, load() raises, draft.py's except-Exception swallows it, and
        # every clip scores a blank final -> CAPPED AT 0.
        # Reproduced: with wifi off, log showed
        #   "[Errno 8] nodename nor servname provided" ... "Retrying in 1s [Retry 1/5]"
        # HF_HUB_OFFLINE=1 forces huggingface_hub to skip the network check
        # entirely and read only from local cache -- which is exactly how the
        # official scoring sandbox will run (network blocked, weights precached
        # by bench/prefetch_models.py). Set here, at call time, so this backend
        # is offline-safe regardless of what the caller's environment has set.
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor,
                                  pipeline)
        self._np = np
        device = os.environ.get("BUILDERR_DEVICE",
                                "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._repo, torch_dtype=dtype, low_cpu_mem_usage=True,
            use_safetensors=True, local_files_only=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(self._repo, local_files_only=True)
        self._pipe = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            torch_dtype=dtype, device=device)
        # warm: compile kernels + fill caches on a throwaway decode.
        self._pipe(np.zeros(16000, dtype=np.float32),
                   generate_kwargs={"task": "transcribe"})

    def transcribe(self, pcm: bytes) -> str:
        audio = self._np.frombuffer(pcm, dtype=self._np.int16).astype(self._np.float32) / 32768.0
        if audio.size == 0:
            return ""
        gk = {"task": "transcribe"}
        lang = os.environ.get("BUILDERR_LANG")
        if lang:
            gk["language"] = lang
        out = self._pipe(audio, generate_kwargs=gk)
        return (out.get("text") or "").strip()


# --------------------------------------------------------------------------
# registry / factory
# --------------------------------------------------------------------------
def available_backends() -> list[str]:
    """Names bench/models.py can sweep. MLX excluded until built.

    Oriserve is registered because it's the primary-model candidate we need to
    measure NOW. If transformers/torch aren't installed it will fail loudly in
    the sweep (caught and reported), which is the correct signal.
    """
    return ["faster-whisper", "oriserve"]


def models_for(backend_name: str) -> list[str]:
    """Each backend iterates over its OWN model identifiers, not a shared list.

    faster-whisper -> whisper sizes; oriserve -> its finetuned variants.
    This is what lets the same sweep compare unlike runtimes fairly.
    """
    if backend_name == "faster-whisper":
        return ["small", "medium", "large-v3-turbo"]
    if backend_name == "oriserve":
        return ["swift", "prime", "apex"]
    if backend_name == "mlx":
        return ["small", "medium", "large-v3-turbo"]
    return ["small"]


def make_backend() -> Backend:
    """Construct the backend selected by env. Default = oriserve/swift.

    CHANGED (backed by Stage-1 benchmark on M1): faster-whisper/small warm-decoded
    at 4554ms (over the 4000ms cap, 6.1/30); oriserve/swift at 1355ms (27.9/30).
    faster-whisper is demonstrably incapable of the latency budget on this
    hardware class, so the shipped default is now the only measured-in-budget
    config. Env vars still override for A/B sweeps.
    """
    kind = os.environ.get("BUILDERR_BACKEND", "oriserve")
    model = os.environ.get("BUILDERR_MODEL", "swift")
    if kind == "faster-whisper":
        return FasterWhisperBackend(
            model,
            compute=os.environ.get("BUILDERR_COMPUTE", "int8"),
            device=os.environ.get("BUILDERR_DEVICE", "cpu"),
        )
    if kind == "oriserve":
        # BUILDERR_MODEL doubles as the variant selector for this backend.
        return OriserveBackend(model if model in OriserveBackend._REPO else "swift")
    if kind == "mlx":
        return MLXWhisperBackend(model)
    raise ValueError(f"unknown BUILDERR_BACKEND={kind!r}")
