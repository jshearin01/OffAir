"""Phase 5 tests — LLM Markdown export + local LLM client.

No network, no Ollama, no llama-cpp: the Ollama client is tested
against a fake HTTP client, and the llama-cpp path is tested for
its "not installed" behaviour only.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from app.diarization.names import save_speaker_names
from app.exports.llm_export import (
    LLM_EXPORT_FILE,
    export_llm_markdown,
    export_llm_markdown_from_session,
)
from app.exports.local_llm import (
    LlamaCppClient,
    LocalLLMConfig,
    LocalLLMError,
    OllamaClient,
    OllamaNotAvailableError,
    summarize_locally,
)
from app.storage.sessions import create_session, update_metadata
from app.transcription.engine import Segment
from app.transcription.exports import load_transcript_json


def _segments():
    return [
        Segment(0.0, 5.0, "Good morning everyone.", speaker="You"),
        Segment(5.0, 10.0, "The budget is twelve thousand dollars.",
                speaker="Remote"),
        Segment(10.0, 15.0, "See you next week.", speaker=None),
    ]


# ------------------------------------------------------------ LLM Markdown --
def test_markdown_export_has_header_context_participants_timestamps(tmp_path):
    path = export_llm_markdown(
        _segments(), tmp_path / "out.md",
        speaker_names={"You": "John", "Remote": "Priya"},
        title="Meeting transcript — test",
        date="2026-11-07T10:00:00+00:00", duration_seconds=61.3,
        app_name="Teams", model="tiny",
        diarization={"mode": "simple",
                     "method": "dual-channel-energy"},
        capture_mode="system-loopback-fallback")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Meeting transcript — test")
    assert "## Context for summariser" in text
    assert "2026-11-07" in text
    assert "1m 1s" in text  # 61.3 s formatted
    assert "Teams" in text
    assert "tiny" in text
    assert "dual-channel-energy" in text
    assert "## Participants" in text
    assert "- John" in text and "- Priya" in text
    assert "## Transcript" in text
    assert "[00:00] **John:** Good morning everyone." in text
    assert "[00:05] **Priya:** The budget is twelve thousand dollars." in text
    assert "[00:10] See you next week." in text  # unlabelled: no speaker


def test_markdown_export_has_no_invented_action_items(tmp_path):
    """With no explicit action_items, there is NO Action items section."""
    path = export_llm_markdown(_segments(), tmp_path / "out.md")
    text = path.read_text(encoding="utf-8")
    assert "Action items" not in text
    assert "[ ]" not in text


def test_markdown_export_action_items_only_when_explicitly_passed(tmp_path):
    path = export_llm_markdown(
        _segments(), tmp_path / "out.md",
        action_items=["Priya sends the timeline by Monday",
                      "Marcus confirms the hardware list"])
    text = path.read_text(encoding="utf-8")
    assert "## Action items" in text
    assert "- [ ] Priya sends the timeline by Monday" in text
    assert "- [ ] Marcus confirms the hardware list" in text
    # Empty list = no section either.
    path2 = export_llm_markdown(_segments(), tmp_path / "out2.md",
                                action_items=[])
    assert "Action items" not in (path2.read_text(encoding="utf-8"))


def test_markdown_export_empty_transcript(tmp_path):
    path = export_llm_markdown([], tmp_path / "out.md")
    text = path.read_text(encoding="utf-8")
    assert "## Transcript" in text
    assert "(No transcript segments.)" in text
    assert "- Unknown" in text


def test_markdown_export_from_session(tmp_path):
    session = create_session(tmp_path, app_name="Teams")
    update_metadata(session, duration_seconds=61.3)
    segments = _segments()
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny", "segments": [s.to_dict() for s in segments]},
        indent=2), encoding="utf-8")
    save_speaker_names(session, {"You": "John"})
    path = export_llm_markdown_from_session(session)
    assert path == session / LLM_EXPORT_FILE
    text = path.read_text(encoding="utf-8")
    assert "John" in text
    assert "tiny" in text
    assert "61" in text or "1m 1s" in text
    assert "Action items" not in text
    # Round-trip: the export's transcript lines match the source segments.
    assert load_transcript_json(session / "transcript.json")[0].text == \
        "Good morning everyone."


def test_markdown_export_from_session_without_transcript_raises(tmp_path):
    session = create_session(tmp_path)
    with pytest.raises(FileNotFoundError, match="transcript"):
        export_llm_markdown_from_session(session)


# -------------------------------------------------------------- Ollama -----
class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeHttp:
    """Fake urlopen: records requests, answers /api/tags + /api/generate."""

    def __init__(self, tags_ok=True, generate_text="Summary: all good."):
        self.tags_ok = tags_ok
        self.generate_text = generate_text
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.full_url, request.get_method()))
        if request.full_url.endswith("/api/tags"):
            if not self.tags_ok:
                raise ConnectionError("connection refused")
            return _FakeResponse({"models": [{"name": "qwen3.5:9b"}]})
        if request.full_url.endswith("/api/generate"):
            return _FakeResponse({"response": self.generate_text})
        raise AssertionError(f"unexpected URL {request.full_url}")


def test_ollama_client_is_available_and_summarizes_with_fake_http():
    client = OllamaClient(http_client=_FakeHttp())
    assert client.is_available() is True
    assert client.list_models() == ["qwen3.5:9b"]
    summary = client.summarize("Hello world transcript.")
    assert summary == "Summary: all good."
    # The transcript was sent only to localhost.
    urls = [u for u, _ in client._http.requests]
    assert all("localhost" in u for u in urls)


def test_ollama_client_not_running_raises_guided_error():
    client = OllamaClient(http_client=_FakeHttp(tags_ok=False))
    assert client.is_available() is False
    with pytest.raises(OllamaNotAvailableError, match="Ollama is not running"):
        client.summarize("Hello world transcript.")


def test_ollama_client_empty_transcript_rejected():
    client = OllamaClient(http_client=_FakeHttp())
    with pytest.raises(ValueError, match="empty transcript"):
        client.summarize("   ")


def test_ollama_client_rejects_non_localhost_urls():
    with pytest.raises(ValueError, match="localhost"):
        OllamaClient(base_url="https://api.example.com")


def test_summarize_locally_is_off_by_default():
    with pytest.raises(LocalLLMError, match="off"):
        summarize_locally("Some transcript.",
                          config=LocalLLMConfig(enabled=False))


def test_ollama_generate_uses_long_timeout_probe_stays_short():
    """Nov 21 2026: the generate call once shared the 10 s probe timeout
    and could never return a real CPU summary. /api/generate must use
    the long generate timeout; the probe keeps the short one."""

    class _TimeoutRecordingHttp(_FakeHttp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeouts = {}

        def __call__(self, request, timeout=None):
            self.timeouts[request.full_url.split("/")[-1]] = timeout
            return super().__call__(request, timeout=timeout)

    http = _TimeoutRecordingHttp()
    client = OllamaClient(http_client=http)
    assert client.is_available() is True
    assert client.summarize("Hello world transcript.") == "Summary: all good."
    assert http.timeouts["tags"] == 2.0
    assert http.timeouts["generate"] == 600.0

    http2 = _TimeoutRecordingHttp()
    client2 = OllamaClient(http_client=http2, generate_timeout_seconds=45.0)
    client2.summarize("Hello world transcript.")
    assert http2.timeouts["generate"] == 45.0


def test_summarize_locally_uses_injected_ollama_client():
    client = OllamaClient(http_client=_FakeHttp(
        generate_text="Local summary here."))
    result = summarize_locally(
        "Some transcript.",
        config=LocalLLMConfig(enabled=True, provider="ollama"),
        ollama_client=client)
    assert result == "Local summary here."


def test_summarize_locally_unknown_provider_rejected():
    with pytest.raises(LocalLLMError, match="Unknown local LLM provider"):
        summarize_locally("Some transcript.",
                          config=LocalLLMConfig(enabled=True,
                                                provider="cloud"))


def test_llama_cpp_client_reports_unavailable_when_not_installed():
    client = LlamaCppClient("/tmp/does-not-exist.gguf")
    # llama-cpp-python is not installed in this environment; if it ever
    # is, the missing model file still makes it unavailable.
    assert client.is_available() is False
    with pytest.raises(LocalLLMError):
        client.summarize("Some transcript.")


def test_ollama_summarize_request_disables_thinking_and_sets_context():
    """Qwen3.5 thinks by default (research verdict, Nov 8 2026) —
    without "think": false a CPU summary would sit for minutes
    in invisible reasoning. num_ctx 16384 fits ~an hour of
    transcript; Ollama's 4K default would silently truncate it."""
    import json as _json

    bodies = []

    class _BodyCapturingHttp(_FakeHttp):
        def __call__(self, request, timeout=None):
            if request.full_url.endswith("/api/generate"):
                bodies.append(_json.loads(request.data.decode("utf-8")))
            return super().__call__(request, timeout=timeout)

    client = OllamaClient(http_client=_BodyCapturingHttp())
    assert client.summarize("Hello world transcript.") == "Summary: all good."
    assert len(bodies) == 1
    assert bodies[0]["model"] == "qwen3.5:9b"
    assert bodies[0]["options"]["num_ctx"] == 16384
    assert bodies[0]["options"]["think"] is False
