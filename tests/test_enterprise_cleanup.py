"""Enterprise-LLM cleanup runner.

The runner shells out to a `claude` CLI on PATH — never ACP/A2A,
never any network call from this code. Every test uses a FAKE
`claude` script on PATH (or none at all): no real CLI, no real
LLM, and no audio ever leaves the machine (only transcript text
is fed on stdin — asserted via the fake's argv sidecar).
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time

import pytest

from app.config import (
    AppConfig,
    config_from_settings,
    load_settings,
    save_settings,
    settings_to_dict,
)
from app.transcription.cleanup import (
    ENTERPRISE_CLEAN_TXT,
    TRANSCRIPT_VARIANT_FILES,
    TRANSCRIPT_VARIANTS,
    transcript_variant_path,
)
from app.transcription.enterprise_cleanup import (
    ENTERPRISE_CLEAN_PROMPT,
    EnterpriseCleanupCancelledError,
    EnterpriseCleanupError,
    EnterpriseCleanupNotAvailableError,
    EnterpriseCleanupRunner,
    EnterpriseCleanupTimeoutError,
    enterprise_clean_session,
    enterprise_clean_text,
)

TRANSCRIPT = ("[00:00] Priya: Um, the {budget} is twelve thousand.\n"
              "[00:05] Marcus: The the deadline is Friday.\n")


def _fake_claude(tmp_path, body, name="claude"):
    """An executable fake `claude` on a PATH dir. `body` is sh code
    run with stdin = the transcript text; $FAKE_ARGS_FILE (when
    set) receives the argv, one per line."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / name
    script.write_text(
        "#!/bin/sh\n"
        'if [ -n "$FAKE_ARGS_FILE" ]; then printf "%s\\n" "$@" '
        '> "$FAKE_ARGS_FILE"; fi\n' + body + "\n",
        encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bindir


def _with_fake_path(monkeypatch, bindir):
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep
                       + os.environ.get("PATH", ""))


def _session_with_transcript(tmp_path):
    session = tmp_path / "2026-11-21_10-00-00_teams"
    session.mkdir(parents=True)
    (session / "transcript.json").write_text(json.dumps(
        {"model": "base", "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "Priya",
             "text": "Um, the budget is twelve thousand."},
            {"start": 5.0, "end": 8.0, "speaker": "Marcus",
             "text": "The the deadline is Friday."}]}, indent=2),
        encoding="utf-8")
    return session


# ------------------------------------------------------------------ success --
def test_success_writes_text_only_artifact_and_leaves_verbatim(
        tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "cat")
    _with_fake_path(monkeypatch, bindir)
    session = _session_with_transcript(tmp_path)
    before = hashlib.sha256(
        (session / "transcript.json").read_bytes()).hexdigest()
    out = enterprise_clean_text(TRANSCRIPT)
    assert out == TRANSCRIPT
    path = enterprise_clean_session(session)
    assert path == session / ENTERPRISE_CLEAN_TXT
    assert path.name == "transcript-enterprise-clean.txt"
    text = path.read_text(encoding="utf-8")
    assert "enterprise" in text.lower()
    assert "audio" in text.lower()  # header states audio never leaves
    assert "Priya: Um, the budget" in text
    # Verbatim untouched, and no segment-shape JSON artifact exists.
    assert hashlib.sha256(
        (session / "transcript.json").read_bytes()).hexdigest() == before
    assert not (session / "transcript-enterprise-clean.json").exists()


def test_prompt_goes_as_dash_p_and_states_deterministic_rules(
        tmp_path, monkeypatch):
    args_file = tmp_path / "argv.txt"
    monkeypatch.setenv("FAKE_ARGS_FILE", str(args_file))
    bindir = _fake_claude(tmp_path, "cat")
    _with_fake_path(monkeypatch, bindir)
    enterprise_clean_text(TRANSCRIPT)
    argv = args_file.read_text(encoding="utf-8").splitlines()
    assert argv[0] == "-p"
    prompt = argv[1]
    assert "filler" in prompt.lower()
    assert "false start" in prompt.lower()
    assert "repeated" in prompt.lower()
    assert "do not add" in prompt.lower()
    assert "speaker label" in prompt.lower()
    assert "timestamp" in prompt.lower()
    # The transcript itself travels on stdin, not in argv.
    assert TRANSCRIPT not in prompt
    # The fixed prompt template carries the same rules.
    assert "{transcript}" in ENTERPRISE_CLEAN_PROMPT


def test_transcript_braces_survive_prompt_substitution(
        tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "cat")
    _with_fake_path(monkeypatch, bindir)
    assert enterprise_clean_text("[00:00] A: {budget} {x}") == \
        "[00:00] A: {budget} {x}"


# ------------------------------------------------------------- missing CLI --
def test_missing_cli_is_a_guided_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()
    with pytest.raises(EnterpriseCleanupNotAvailableError) as excinfo:
        enterprise_clean_text(TRANSCRIPT,
                              claude_command="claude-definitely-absent")
    message = str(excinfo.value)
    assert "claude" in message.lower()
    assert "install" in message.lower()


def test_missing_configured_command_path_is_guided(tmp_path):
    with pytest.raises(EnterpriseCleanupNotAvailableError) as excinfo:
        enterprise_clean_text(
            TRANSCRIPT, claude_command="/no/such/dir/claude")
    assert "claude" in str(excinfo.value).lower()


# ------------------------------------------------------------- timeout/cancel --
def test_timeout_kills_process_and_writes_no_artifact(tmp_path,
                                                      monkeypatch):
    marker = tmp_path / "finished.marker"
    bindir = _fake_claude(
        tmp_path, f"sleep 5; touch {marker}; echo done")
    _with_fake_path(monkeypatch, bindir)
    session = _session_with_transcript(tmp_path)
    started = time.monotonic()
    with pytest.raises(EnterpriseCleanupTimeoutError):
        enterprise_clean_text(TRANSCRIPT, timeout=0.3)
    assert time.monotonic() - started < 3.0
    assert not marker.exists()  # killed before it could finish
    assert not (session / ENTERPRISE_CLEAN_TXT).exists()
    # The session-level entry also leaves no artifact on timeout.
    with pytest.raises(EnterpriseCleanupTimeoutError):
        enterprise_clean_session(session, timeout=0.3)
    assert not (session / ENTERPRISE_CLEAN_TXT).exists()


def test_cancel_via_predicate_writes_no_artifact(tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "sleep 5; echo done")
    _with_fake_path(monkeypatch, bindir)
    cancelled = threading.Event()

    def _cancel_soon():
        time.sleep(0.2)
        cancelled.set()

    threading.Thread(target=_cancel_soon, daemon=True).start()
    with pytest.raises(EnterpriseCleanupCancelledError):
        enterprise_clean_text(TRANSCRIPT, timeout=30.0,
                              is_cancelled=cancelled.is_set)


def test_cancel_via_runner_cancel_kills_process(tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "sleep 5; echo done")
    _with_fake_path(monkeypatch, bindir)
    runner = EnterpriseCleanupRunner(timeout=30.0)

    def _cancel_soon():
        time.sleep(0.2)
        runner.cancel()

    threading.Thread(target=_cancel_soon, daemon=True).start()
    with pytest.raises(EnterpriseCleanupCancelledError):
        runner.run(TRANSCRIPT)


# ------------------------------------------------------------- other failures --
def test_nonzero_exit_and_empty_output_are_errors_no_artifact(
        tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "echo 'bad prompt' >&2; exit 3")
    _with_fake_path(monkeypatch, bindir)
    with pytest.raises(EnterpriseCleanupError, match="exit code 3"):
        enterprise_clean_text(TRANSCRIPT)
    bindir2 = _fake_claude(tmp_path, "true", name="claude")
    _with_fake_path(monkeypatch, bindir2)
    with pytest.raises(EnterpriseCleanupError, match="no output"):
        enterprise_clean_text(TRANSCRIPT)


def test_blank_transcript_rejected_before_spawning(tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "cat")
    _with_fake_path(monkeypatch, bindir)
    with pytest.raises(ValueError, match="empty"):
        enterprise_clean_text("   ")


def test_session_without_transcript_raises(tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "cat")
    _with_fake_path(monkeypatch, bindir)
    session = tmp_path / "empty-session"
    session.mkdir()
    with pytest.raises(FileNotFoundError):
        enterprise_clean_session(session)


# ------------------------------------------------------------- config gating --
def test_enterprise_cleanup_disabled_by_default_and_config_round_trips(
        tmp_path):
    fresh = AppConfig()
    assert fresh.enterprise_cleanup_enabled is False
    assert fresh.enterprise_cleanup_timeout_seconds == 120.0
    assert fresh.enterprise_cleanup_claude_command == "claude"
    path = tmp_path / "settings.json"
    save_settings(AppConfig(
        enterprise_cleanup_enabled=True,
        enterprise_cleanup_timeout_seconds=60.0,
        enterprise_cleanup_claude_command="/usr/local/bin/claude"), path)
    loaded = load_settings(path)
    assert loaded.enterprise_cleanup_enabled is True
    assert loaded.enterprise_cleanup_timeout_seconds == 60.0
    assert loaded.enterprise_cleanup_claude_command == "/usr/local/bin/claude"
    data = settings_to_dict(loaded)
    assert data["enterprise_cleanup_enabled"] is True
    assert "ollama_model" in data  # other settings untouched
    # Invalid values fall back to defaults, one key at a time.
    assert config_from_settings(
        {"enterprise_cleanup_timeout_seconds": -5}
    ).enterprise_cleanup_timeout_seconds == 120.0
    assert config_from_settings(
        {"enterprise_cleanup_claude_command": "  "}
    ).enterprise_cleanup_claude_command == "claude"


def test_session_run_refuses_when_config_disables_it(tmp_path, monkeypatch):
    bindir = _fake_claude(tmp_path, "cat")
    _with_fake_path(monkeypatch, bindir)
    session = _session_with_transcript(tmp_path)
    with pytest.raises(EnterpriseCleanupError, match="off by default"):
        enterprise_clean_session(session, config=AppConfig())
    assert not (session / ENTERPRISE_CLEAN_TXT).exists()


# ------------------------------------------------------------- variant registry --
def test_third_variant_registered_in_cleanup(tmp_path):
    assert TRANSCRIPT_VARIANTS == ("original", "cleaned", "enterprise")
    assert TRANSCRIPT_VARIANT_FILES["enterprise"] == ENTERPRISE_CLEAN_TXT
    assert transcript_variant_path(tmp_path, "enterprise") == (
        tmp_path / ENTERPRISE_CLEAN_TXT)
    assert transcript_variant_path(tmp_path, "cleaned").name == \
        "transcript-clean.txt"
    with pytest.raises(ValueError, match="unknown transcript variant"):
        transcript_variant_path(tmp_path, "polished")
