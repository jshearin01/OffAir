"""Enterprise-LLM transcript cleanup via the Claude Code CLI.

Opt-in, OFF by default. The cleanup runs the headless
``claude -p`` subprocess — subprocess ONLY, never ACP, never A2A,
never a network call from this code. TRANSCRIPT TEXT ONLY leaves
the machine: the verbatim transcript text is fed to the CLI on
stdin, and the cleaned text comes back on stdout. Audio NEVER
leaves the machine — no audio path is passed, read, or uploaded
by anything in this module. The verbatim transcript is never
modified; the result is a third, distinct artifact,
``transcript-enterprise-clean.txt`` (text only — the LLM returns
unstructured text, so no segment-shape ``.json`` is written; see
:mod:`app.transcription.cleanup` for the variant registration).

The runner is worker-thread-safe: it holds no module-global
state, each call spawns its own process, and cancellation kills
that process. Call it from a worker thread, never the UI thread
— a run can take up to the timeout. A timeout, a cancel, a
non-zero exit, or empty output all raise and write NO artifact:
no partial result is ever presented as complete, and a failed
run leaves any earlier successful artifact untouched.

When the ``claude`` CLI is absent, the error names the install
instead of failing cryptically.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path

from app.diarization.names import load_speaker_names
from app.transcription.cleanup import ENTERPRISE_CLEAN_TXT
from app.transcription.engine import TRANSCRIPT_JSON
from app.transcription.exports import (
    format_txt_timestamp,
    load_transcript_json,
)

logger = logging.getLogger("offair.transcription.enterprise_cleanup")

#: Default timeout (seconds) — matches
#: ``app.config.ENTERPRISE_CLEANUP_DEFAULT_TIMEOUT_SECONDS``.
DEFAULT_TIMEOUT_SECONDS: float = 120.0
#: Default CLI command — matches
#: ``app.config.ENTERPRISE_CLEANUP_DEFAULT_COMMAND``.
DEFAULT_CLAUDE_COMMAND: str = "claude"

#: The fixed prompt. It states its own instruction set — filler
#: removal, false-start repair, repeated-word dedup, punctuation
#: tidy, keeping every speaker label and timestamp, adding
#: nothing, and removing nothing spoken except fillers and false
#: starts. This is NOT the deterministic layer's rule set: the
#: regex false-start/repeat rules were removed Nov 22 2026 (see
#: docs/notes/cleanup-repeat-rule-2026-11-22.md), but the prompt
#: keeps asking for that handling because an LLM judging intent
#: from context is a different mechanism from the removed
#: regexes. Dictionary-style name sense is NOT available here
#: (the prompt carries no dictionary).
#: ``{transcript}`` is replaced literally (never ``str.format`` —
#: transcript text may contain braces).
ENTERPRISE_CLEAN_PROMPT = (
    "You are cleaning up a meeting transcript. Rewrite the transcript "
    "below to fix grammar, punctuation and obvious transcription "
    "errors, remove filler words (um, uh, hmm, er, erm, mmm, ah), "
    "repair false starts (a repeated 1-3 word phrase followed by "
    "more words), remove repeated words, and tidy punctuation and "
    "spacing. Keep every speaker label and timestamp exactly as "
    "written. Do not add facts, names, dates or commitments that "
    "are not in the transcript, and do not remove any spoken "
    "content except fillers and false starts. Return only the "
    "rewritten transcript, in the same [MM:SS] Speaker: text "
    "format.\n\nTranscript:\n{transcript}"
)

ENTERPRISE_CLEAN_HEADER = (
    "# Transcript (enterprise-cleaned)\n\n"
    "> This is a cleanup of the verbatim transcript by an external "
    "LLM (Claude Code CLI, `claude -p`). Only the transcript TEXT "
    "was sent to the CLI — no audio ever leaves this machine. It "
    "may contain errors: the verbatim transcript in "
    "transcript.json / transcript.txt is the record. This file is "
    "a readability aid only, never a replacement for the verbatim "
    "transcript.\n\n"
)

_INSTALL_GUIDANCE = (
    "The Claude Code CLI (`claude`) is not installed or not on PATH. "
    "Install Claude Code from https://claude.com/product/claude-code "
    "(or `npm install -g @anthropic-ai/claude-code`) and make sure "
    "`claude` is on PATH, then try again. The enterprise cleanup "
    "sends transcript text only — no audio — to the CLI."
)


class EnterpriseCleanupError(Exception):
    """The enterprise cleanup failed (no artifact was written)."""


class EnterpriseCleanupNotAvailableError(EnterpriseCleanupError):
    """The `claude` CLI is missing — the error names the install."""


class EnterpriseCleanupTimeoutError(EnterpriseCleanupError):
    """The CLI did not finish within the timeout (process killed)."""


class EnterpriseCleanupCancelledError(EnterpriseCleanupError):
    """The run was cancelled (process killed)."""


def _resolve_command(claude_command: str) -> str:
    """Executable path for the CLI, or raise the guided error."""
    command = str(claude_command).strip()
    if not command:
        raise EnterpriseCleanupNotAvailableError(_INSTALL_GUIDANCE)
    if "/" in command or "\\" in command or os.path.sep in command:
        path = Path(command)
        if path.is_file():
            return str(path)
        raise EnterpriseCleanupNotAvailableError(
            f"The configured Claude Code command {command!r} was not "
            f"found. {_INSTALL_GUIDANCE}")
    found = shutil.which(command)
    if found is None:
        raise EnterpriseCleanupNotAvailableError(_INSTALL_GUIDANCE)
    return found


def _prompt_for(transcript_text: str) -> str:
    return ENTERPRISE_CLEAN_PROMPT.replace("{transcript}",
                                           transcript_text)


def _kill_process(process: subprocess.Popen) -> None:
    """Kill a CLI process and any children it spawned.

    The child runs in its own process group (see :meth:`run`),
    so killing the group takes down grandchildren too — killing
    only the direct child leaves a grandchild holding the stdout
    pipe open, and ``communicate`` then blocks until that child
    exits on its own (measured: a 5 s ``sleep`` grandchild made
    a 0.3 s timeout take 5 s). On Windows there is no process
    group to kill, so the process itself is killed.
    """
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
            return
    except (ProcessLookupError, OSError):
        pass
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass


class EnterpriseCleanupRunner:
    """One enterprise-cleanup runner: subprocess-only, cancellable.

    Safe to call from a worker thread. :meth:`cancel` (from any
    thread, e.g. the UI thread while :meth:`run` executes in a
    worker) sets the cancel flag and kills the running process.
    A runner holds no state between runs except the cancel flag,
    which :meth:`run` clears at entry — reuse across runs is safe,
    and separate runners are fully independent.
    """

    def __init__(self, *, claude_command: str = DEFAULT_CLAUDE_COMMAND,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        command = str(claude_command).strip()
        if not command:
            raise ValueError("claude command must not be blank")
        timeout_value = float(timeout)
        if not timeout_value > 0:
            raise ValueError("timeout must be positive")
        self.claude_command = command
        self.timeout = timeout_value
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._process: subprocess.Popen | None = None

    def cancel(self) -> None:
        """Cancel a running cleanup: flag it, kill the process."""
        self._cancel_event.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            _kill_process(process)

    def run(self, transcript_text: str, *,
            is_cancelled=None) -> str:
        """Run the CLI over ``transcript_text``; return cleaned text.

        ``is_cancelled`` is an optional zero-argument predicate
        checked while the process runs (the same convention as
        :meth:`app.transcription.engine.TranscriptionEngine.
        transcribe_file`). Raises
        :class:`EnterpriseCleanupNotAvailableError` when the CLI
        is missing, :class:`EnterpriseCleanupTimeoutError` on
        timeout, :class:`EnterpriseCleanupCancelledError` on
        cancel, and :class:`EnterpriseCleanupError` on a non-zero
        exit or empty output. Never returns a partial result.
        """
        if not transcript_text or not str(transcript_text).strip():
            raise ValueError("Cannot clean an empty transcript")
        self._cancel_event.clear()
        executable = _resolve_command(self.claude_command)
        prompt = _prompt_for(str(transcript_text))
        popen_kwargs: dict = {
            "stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE, "text": True,
        }
        if os.name == "posix":
            # Own process group, so a cancel/timeout kill takes
            # down the whole tree (see _kill_process).
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen([executable, "-p", prompt],
                                   **popen_kwargs)
        with self._lock:
            self._process = process
        result: dict = {}

        def _communicate() -> None:
            try:
                out, err = process.communicate(
                    input=str(transcript_text))
                result["stdout"] = out
                result["stderr"] = err
                result["returncode"] = process.returncode
            except Exception as exc:  # noqa: BLE001 — reported below
                result["error"] = exc

        worker = threading.Thread(target=_communicate, daemon=True)
        worker.start()
        try:
            import time

            deadline_set_at = time.monotonic() + self.timeout
            while worker.is_alive():
                if self._cancel_event.is_set() or (
                        is_cancelled is not None and is_cancelled()):
                    self.cancel()
                    worker.join(timeout=5.0)
                    raise EnterpriseCleanupCancelledError(
                        "Enterprise cleanup was cancelled — no "
                        "artifact was written.")
                if time.monotonic() >= deadline_set_at:
                    self.cancel()
                    worker.join(timeout=5.0)
                    raise EnterpriseCleanupTimeoutError(
                        f"Enterprise cleanup timed out after "
                        f"{self.timeout:g}s — no artifact was written.")
                worker.join(timeout=0.05)
        finally:
            if process.poll() is None:
                _kill_process(process)
            with self._lock:
                if self._process is process:
                    self._process = None
            worker.join(timeout=5.0)
        # Race: cancel() may have killed the process just as the
        # worker thread finished, so the poll loop never saw the
        # flag and the result looks like a plain exit-code failure
        # (-9). A cancel that arrived during the run wins over the
        # exit code.
        if self._cancel_event.is_set() or (
                is_cancelled is not None and is_cancelled()):
            raise EnterpriseCleanupCancelledError(
                "Enterprise cleanup was cancelled — no artifact "
                "was written.")
        if "error" in result:
            raise EnterpriseCleanupError(
                f"Enterprise cleanup failed: {result['error']}")
        returncode = result.get("returncode")
        if returncode != 0:
            stderr = str(result.get("stderr") or "").strip()
            detail = f": {stderr[-500:]}" if stderr else ""
            raise EnterpriseCleanupError(
                f"Enterprise cleanup failed (exit code {returncode})"
                f"{detail} — no artifact was written.")
        cleaned = str(result.get("stdout") or "")
        if not cleaned.strip():
            raise EnterpriseCleanupError(
                "Enterprise cleanup produced no output — no artifact "
                "was written.")
        return cleaned


def enterprise_clean_text(transcript_text: str, *,
                          claude_command: str = DEFAULT_CLAUDE_COMMAND,
                          timeout: float = DEFAULT_TIMEOUT_SECONDS,
                          is_cancelled=None) -> str:
    """One-shot enterprise cleanup (a fresh runner per call)."""
    runner = EnterpriseCleanupRunner(claude_command=claude_command,
                                     timeout=timeout)
    return runner.run(transcript_text, is_cancelled=is_cancelled)


def _verbatim_text(session_dir: Path) -> str:
    """Verbatim transcript as ``[MM:SS] Speaker: text`` lines.

    Built from transcript.json (speaker display names applied),
    so it does not depend on transcript.txt being in any
    particular state. This is the text — and only the text —
    sent to the CLI.
    """
    transcript_path = session_dir / TRANSCRIPT_JSON
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"No {TRANSCRIPT_JSON} in {session_dir} — transcribe the "
            "session first.")
    segments = load_transcript_json(transcript_path)
    if not segments:
        raise ValueError("Cannot clean an empty transcript")
    try:
        names = load_speaker_names(session_dir)
    except ValueError:
        names = {}
    lines = []
    for seg in segments:
        speaker = getattr(seg, "speaker", None)
        display = names.get(speaker, speaker) if speaker else None
        prefix = f"{display}: " if display else ""
        lines.append(f"{format_txt_timestamp(seg.start)} "
                     f"{prefix}{seg.text}")
    return "\n".join(lines) + "\n"


def enterprise_clean_session(session_dir: str | Path, *,
                             config=None,
                             claude_command: str | None = None,
                             timeout: float | None = None,
                             is_cancelled=None,
                             runner: EnterpriseCleanupRunner | None = None
                             ) -> Path:
    """Enterprise-clean a session's verbatim transcript.

    Writes ``transcript-enterprise-clean.txt`` (header + cleaned
    text) only after the CLI run succeeds in full; the verbatim
    files are only read. ``config`` is an
    :class:`app.config.AppConfig`: when given and
    ``enterprise_cleanup_enabled`` is False, the run is refused
    (opt-in, off by default) — pass no config for an explicit
    direct call, or a config with the setting enabled. Explicit
    ``claude_command`` / ``timeout`` arguments win over the
    config; ``runner`` (when given) wins over both for execution.
    Raises the same errors as :meth:`EnterpriseCleanupRunner.run`
    and writes nothing on failure.
    """
    session_dir = Path(session_dir)
    if config is not None and not config.enterprise_cleanup_enabled:
        raise EnterpriseCleanupError(
            "Enterprise cleanup is off by default (opt-in) — enable "
            "it in Settings before running it. No artifact was "
            "written.")
    if runner is None:
        command = (claude_command if claude_command is not None
                   else (config.enterprise_cleanup_claude_command
                         if config is not None
                         else DEFAULT_CLAUDE_COMMAND))
        timeout_value = (timeout if timeout is not None
                         else (config.enterprise_cleanup_timeout_seconds
                               if config is not None
                               else DEFAULT_TIMEOUT_SECONDS))
        runner = EnterpriseCleanupRunner(claude_command=command,
                                         timeout=timeout_value)
    text = _verbatim_text(session_dir)
    cleaned = runner.run(text, is_cancelled=is_cancelled)
    target = session_dir / ENTERPRISE_CLEAN_TXT
    tmp = session_dir / (ENTERPRISE_CLEAN_TXT + ".tmp")
    tmp.write_text(ENTERPRISE_CLEAN_HEADER + cleaned.strip() + "\n",
                   encoding="utf-8")
    os.replace(tmp, target)
    logger.info("Wrote enterprise-cleaned transcript to %s", target)
    return target


__all__ = [
    "DEFAULT_CLAUDE_COMMAND",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENTERPRISE_CLEAN_HEADER",
    "ENTERPRISE_CLEAN_PROMPT",
    "EnterpriseCleanupCancelledError",
    "EnterpriseCleanupError",
    "EnterpriseCleanupNotAvailableError",
    "EnterpriseCleanupRunner",
    "EnterpriseCleanupTimeoutError",
    "enterprise_clean_session",
    "enterprise_clean_text",
]
