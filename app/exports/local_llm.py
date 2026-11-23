"""Optional local LLM summary / action items (PRD Phase 5).

OFF by default, local only, never a cloud API:

* **Ollama** — HTTP to ``http://localhost:11434`` (the Ollama
  server running on the same machine). If Ollama is not running,
  :meth:`OllamaClient.is_available` returns False and summarising
  raises :class:`OllamaNotAvailableError` with install/start
  instructions instead of failing obscurely.
* **llama-cpp-python** — loaded lazily, only if installed. Importing
  this module never imports it.

The only URLs ever contacted are localhost ones: the constructor
rejects any base URL whose host is not ``localhost`` or
``127.0.0.1``. No transcript text is ever sent anywhere else.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from app.config import OLLAMA_DEFAULT_MODEL as DEFAULT_OLLAMA_MODEL

logger = logging.getLogger("offair.exports.local_llm")

DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: Timeout for the /api/generate call (Nov 21 2026): a CPU summary
#: takes minutes, not seconds — the old single 10 s timeout (shared
#: with the /api/tags probe) would have killed every real CPU
#: summary with a timeout error before the model answered. The
#: availability probe and model listing keep the short timeout;
#: only generation gets the long one. Found by code reading during
#: the Ollama proof work — no real Ollama summary has run on this
#: VM yet, so this value is unproven on hardware (see
#: docs/notes/ollama-proof-2026-11-21.md).
OLLAMA_GENERATE_TIMEOUT_SECONDS: float = 600.0

_SUMMARY_PROMPT = (
    "You are summarising a meeting transcript. Summarise the meeting "
    "in a few short paragraphs, then list concrete action items as "
    "bullet points, each with the person responsible if the transcript "
    "says so. Only use facts that appear in the transcript — do not "
    "invent names, dates, or commitments.\n\nTranscript:\n{transcript}"
)


class LocalLLMError(RuntimeError):
    """A local LLM operation failed (not configured / not available)."""


class OllamaNotAvailableError(LocalLLMError):
    """Ollama is not reachable on localhost — it is not running."""


def _require_localhost(url: str) -> str:
    """Reject any URL that is not localhost. Returns the URL unchanged."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(
            f"OffAir only talks to a local LLM on localhost, got "
            f"{url!r} — cloud LLMs are never used.")
    return url


@dataclass
class LocalLLMConfig:
    """Local-LLM settings. ``enabled`` is False by default (PRD Phase 5)."""

    enabled: bool = False
    provider: str = "ollama"  # "ollama" or "llama-cpp"
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    llama_cpp_model_path: str | None = None


class OllamaClient:
    """Minimal Ollama client (stdlib only — no ``requests`` dependency).

    ``http_client`` is injectable for tests: an object with
    ``urlopen(request, timeout=...)`` returning a response with
    ``read()`` and a context-manager interface, like
    :func:`urllib.request.urlopen`.
    """

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL,
                 model: str = DEFAULT_OLLAMA_MODEL,
                 timeout_seconds: float = 10.0,
                 generate_timeout_seconds: float
                 = OLLAMA_GENERATE_TIMEOUT_SECONDS,
                 http_client=None) -> None:
        self.base_url = _require_localhost(base_url).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.generate_timeout_seconds = generate_timeout_seconds
        self._http = http_client or urllib.request.urlopen

    # ------------------------------------------------------------ status ---
    def is_available(self) -> bool:
        """True if an Ollama server answers on localhost (fast probe)."""
        try:
            request = urllib.request.Request(
                f"{self.base_url}/api/tags", method="GET")
            with self._http(request, timeout=2.0) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001 — any failure means "not running"
            return False

    def list_models(self) -> list[str]:
        """Model names the local Ollama server reports, or []."""
        try:
            request = urllib.request.Request(
                f"{self.base_url}/api/tags", method="GET")
            with self._http(request, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in payload.get("models", [])
                    if m.get("name")]
        except Exception:  # noqa: BLE001 — surfaced as empty, logged
            logger.warning("Could not list Ollama models", exc_info=True)
            return []

    # --------------------------------------------------------- summarise ---
    def summarize(self, transcript_text: str, *,
                  prompt_template: str | None = None) -> str:
        """Summarise ``transcript_text`` with the local Ollama model.

        Raises OllamaNotAvailableError if the server does not answer,
        LocalLLMError on any other failure. The transcript is sent
        only to localhost.
        """
        if not transcript_text.strip():
            raise ValueError("Cannot summarise an empty transcript")
        if not self.is_available():
            raise OllamaNotAvailableError(
                "Ollama is not running on localhost:11434. Start it "
                "(install from https://ollama.com, then `ollama serve`, "
                "and `ollama pull qwen3.5:9b`), or leave local-LLM "
                "summaries off and paste the Markdown export into an "
                "LLM yourself.")
        prompt = (prompt_template or _SUMMARY_PROMPT).format(
            transcript=transcript_text)
        # num_ctx 16384: Ollama's default context is 4K on low-memory
        # machines, and a 1-hour meeting transcript is ~10-15K
        # tokens — without this the tail of a long meeting is
        # silently truncated. 16384 costs ~512 MB of KV cache at
        # 9B, and fits the default model's 256K window easily.
        # think False: Qwen3.5 reasons ("thinks") by default, which
        # on CPU means minutes of invisible thinking tokens before
        # any summary appears — summarization needs the answer,
        # not the chain of thought.
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"num_ctx": 16384, "think": False},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with self._http(request,
                            timeout=self.generate_timeout_seconds
                            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LocalLLMError(
                f"Ollama returned HTTP {exc.code} — is model "
                f"{self.model!r} pulled? Try `ollama pull {self.model}`."
            ) from exc
        except Exception as exc:  # noqa: BLE001 — wrapped for the UI
            raise LocalLLMError(f"Ollama request failed: {exc}") from exc
        text = str(payload.get("response", "")).strip()
        if not text:
            raise LocalLLMError("Ollama returned an empty response")
        return text


class LlamaCppClient:
    """llama-cpp-python summariser — lazy, optional, local only.

    Constructing this class never imports ``llama_cpp``; the import
    happens on first use, and a clear error is raised if the package
    is not installed. No model is ever downloaded by this code — the
    caller supplies a local GGUF file path.
    """

    def __init__(self, model_path: str, *, n_ctx: int = 4096) -> None:
        if not model_path:
            raise ValueError("llama-cpp needs a local model file path")
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self._llm = None

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        from pathlib import Path
        return Path(self.model_path).exists()

    def _load(self):
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise LocalLLMError(
                "llama-cpp-python is not installed — install it with "
                "`pip install llama-cpp-python` (CPU-only), or use "
                "Ollama instead.") from exc
        from pathlib import Path
        if not Path(self.model_path).exists():
            raise LocalLLMError(
                f"llama-cpp model file not found: {self.model_path}")
        self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                          verbose=False)
        return self._llm

    def summarize(self, transcript_text: str, *,
                  prompt_template: str | None = None) -> str:
        if not transcript_text.strip():
            raise ValueError("Cannot summarise an empty transcript")
        llm = self._load()
        prompt = (prompt_template or _SUMMARY_PROMPT).format(
            transcript=transcript_text)
        result = llm(prompt, max_tokens=1024, stop=["Transcript:"])
        text = str(result["choices"][0]["text"]).strip()
        if not text:
            raise LocalLLMError("llama-cpp returned an empty response")
        return text


def summarize_locally(transcript_text: str, *,
                      config: LocalLLMConfig | None = None,
                      ollama_client: OllamaClient | None = None,
                      llama_cpp_client: LlamaCppClient | None = None) -> str:
    """Summarise with the configured local LLM (default: off).

    Raises LocalLLMError when local LLM is disabled, not configured,
    or not reachable — the caller (UI) turns that into a friendly
    message. Never falls back to a cloud service.
    """
    cfg = config or LocalLLMConfig()
    if not cfg.enabled:
        raise LocalLLMError(
            "Local LLM summaries are off. Enable them in Settings "
            "(Ollama on localhost, or llama-cpp with a local model "
            "file), or paste the Markdown export into an LLM yourself.")
    if cfg.provider == "ollama":
        client = ollama_client or OllamaClient(
            base_url=cfg.ollama_base_url, model=cfg.ollama_model)
        return client.summarize(transcript_text)
    if cfg.provider == "llama-cpp":
        if llama_cpp_client is None and not cfg.llama_cpp_model_path:
            raise LocalLLMError(
                "llama-cpp needs a local model file path — set it in "
                "Settings.")
        client = llama_cpp_client or LlamaCppClient(
            cfg.llama_cpp_model_path or "")
        return client.summarize(transcript_text)
    raise LocalLLMError(f"Unknown local LLM provider {cfg.provider!r}")
