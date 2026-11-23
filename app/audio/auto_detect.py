"""Auto-detect + auto-record: start recording when a Teams meeting starts.

Watches the target app's (Teams by default) audio sessions on Windows
and fires ``on_meeting_started`` / ``on_meeting_ended`` callbacks on a
background thread (1 Hz polling). The main window subscribes to those
events and drives the existing :class:`app.audio.capture.DualCapture`
path — the watcher itself never touches audio hardware.

Gates (from ``docs/PER_APP_CAPTURE.md``, "Auto-detect + auto-record"
design section — implemented Nov 8 2026):

* **Start gate — BOTH required.** The target's audio session is
  **Active** *and* its peak level stays above
  :data:`app.config.AUTO_DETECT_PEAK_THRESHOLD` for a sustained
  :data:`app.config.AUTO_DETECT_SUSTAIN_SECONDS` (2–3 s debounce).
  "Process is running" alone never starts a recording — Teams runs
  all day. A notification blip or a ringtone burst is shorter than
  the debounce, so it never starts one either.
* **Stop gate — ANY (revised Nov 8 2026).** Silence mid-meeting
  NEVER stops a recording — real meetings pause longer than 20 s
  (silent presenting, thinking, reading), and the old 20 s silence
  rule would have cut those off. Stop only on: (a) the session
  going Inactive/Expired and STAYING that way for
  :data:`app.config.AUTO_DETECT_INACTIVE_GRACE_SECONDS` (60 s
  grace — a single inactive poll never stops anything; the grace
  covers transient stream renegotiation mid-call); (b) the target
  process exiting (immediate); or (c) the long-silence fuse —
  session Active with peak at/below
  :data:`app.config.AUTO_DETECT_SILENCE_PEAK` **and** the mic
  unclaimed, continuously for
  :data:`app.config.AUTO_DETECT_SILENCE_FUSE_SECONDS` (10 min —
  a stuck/leftover session, not a real pause). The gate rests on
  **session state** being the meeting-in-progress signal: Teams
  holds its audio session Active for the whole call (it keeps
  rendering, even silence, through the open stream), so peak level
  is only the start debounce. That assumption MUST be verified on
  a real Windows Teams call — watch session state across a 60+ s
  silent stretch; if Teams actually closes the session between
  utterances, this design needs revisiting (see
  ``docs/PER_APP_CAPTURE.md``).
* **Secondary start rule.** Session Active for
  :data:`app.config.AUTO_DETECT_ACTIVE_MIC_SECONDS` (> 30 s) with
  the mic claimed counts as a meeting even with no peak — the
  false-negative mitigation for an all-muted meeting. Mic-claim is
  a corroborating vote only, never the sole signal. Residual gap
  (documented): an all-muted meeting where the mic is never claimed
  will not auto-start.
* **Manual-stop suppression.** After the user stops a recording by
  hand, the watcher does not re-trigger on the still-active session
  — it waits until the session goes inactive/absent first (see
  :meth:`MeetingWatcher.notify_manual_stop`).

Session discovery enumerates **all render endpoints** via
``IMMDeviceEnumerator`` — pycaw's ``GetAllSessions()`` covers the
default endpoint only and would miss Teams on a Bluetooth headset
or a second output device. The mic-claim check enumerates the
capture endpoints the same way.

Platform: **Windows only.** The session-enumeration layer is WASAPI
(pycaw + comtypes). On any other platform
:func:`auto_detect_available` returns ``(False, reason)`` and the
watcher refuses to start — the Settings UI says so instead of
pretending it works. No fake Linux detection is attempted.

Pre-roll (Nov 21 2026): :class:`MeetingWatcher` accepts an
optional :class:`app.audio.preroll.PreRollBuffer` and drains it
when a meeting starts; the UI layer takes the drained audio via
:meth:`MeetingWatcher.take_preroll` and hands it to capture as
``preroll_samples``. The always-on feeder that fills the buffer
is :class:`app.audio.preroll_feeder.PreRollFeeder` (driven by the
UI layer: ``feeder.stop()`` before ``take_preroll()`` and capture
start — see ``app/audio/preroll_feeder.py`` for the contract).
The buffer is memory-only and is drained only on a start, never
written to disk by this module.

What is **not** verified here: no real Windows run exists yet — this
Linux VM has no Windows audio stack, so the COM probe
(:func:`_probe_windows`) is exercised only through injected fake
probes (see ``tests/test_auto_detect.py``). The first real Windows
run — a real Teams meeting start/stop — is the open gate; see
``docs/PER_APP_CAPTURE.md`` for the exact test procedure. In
particular the revised stop gate assumes Teams keeps its audio
session Active across quiet stretches (60+ s of silence) — that
must be watched live on a real call; if Teams closes the session
between utterances, the design needs revisiting.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass

from app.audio.app_sessions import (
    CLSCTX_ALL,
    CLSID_MMDEVICE_ENUMERATOR,
    DEVICE_STATE_ACTIVE,
    EDATAFLOW_CAPTURE,
    EDATAFLOW_RENDER,
    SESSION_STATE_ACTIVE,
    SESSION_STATE_EXPIRED,
    SESSION_STATE_INACTIVE,
    _enumerate_endpoints,
    _read_session,
    _session_manager_for,
)
from app.config import (
    AUTO_DETECT_ACTIVE_MIC_SECONDS,
    AUTO_DETECT_INACTIVE_GRACE_SECONDS,
    AUTO_DETECT_PEAK_THRESHOLD,
    AUTO_DETECT_POLL_INTERVAL_SECONDS,
    AUTO_DETECT_SILENCE_FUSE_SECONDS,
    AUTO_DETECT_SILENCE_PEAK,
    AUTO_DETECT_SUSTAIN_SECONDS,
    TEAMS_PROCESS_NAMES,
)

logger = logging.getLogger(__name__)

#: ``RPC_E_CHANGED_MODE`` — CoInitializeEx on an already-initialised
#: thread; safe to continue (the session interfaces are apartment
#: neutral for the read-only calls used here).
RPC_E_CHANGED_MODE = 0x80010106


@dataclass
class SessionSnapshot:
    """One poll's view of the target app's audio activity.

    ``session_state`` is ``"active"`` / ``"inactive"`` / ``"expired"``
    for the target's render session, or ``None`` when no session for
    the target exists right now. ``peak`` is the session's current
    peak level (0.0–1.0) or ``None`` when it could not be read.
    ``mic_claimed`` is True when the target has an active session on
    a capture endpoint (it is speaking / has the mic). ``probe_error``
    is set when enumeration itself failed — the watcher skips that
    tick instead of treating a COM hiccup as "meeting ended".
    """

    process_running: bool = False
    session_state: str | None = None
    peak: float | None = None
    mic_claimed: bool = False
    probe_error: str | None = None


def auto_detect_available() -> tuple[bool, str | None]:
    """Is meeting auto-detect usable on this machine?

    Returns ``(True, None)`` or ``(False, reason)``. Never raises.
    Needs Windows (WASAPI session enumeration) + pycaw. Unlike
    per-app capture it does **not** need Windows build 20348 —
    session enumeration predates process loopback.
    """
    if sys.platform != "win32":
        return False, (
            "Auto-detect needs Windows (WASAPI audio sessions); "
            f"this machine is {sys.platform}. Meeting auto-detect "
            "is not available here — record manually instead.")
    try:
        import pycaw  # noqa: F401  # type: ignore[import-not-found]
        import comtypes  # noqa: F401
    except Exception as exc:  # pragma: no cover - Windows-only branch
        return False, (
            "Auto-detect needs the 'pycaw' and 'comtypes' packages "
            f"({exc}); install them with "
            "`pip install -r requirements.txt`.")
    return True, None


# --------------------------------------------------------------------------
# Real Windows probe (lazy COM — the module imports on Linux)
# --------------------------------------------------------------------------
def _probe_windows(process_names: list[str]) -> SessionSnapshot:
    """Probe the target app's audio activity via WASAPI sessions.

    Assumes COM is initialised on the calling thread (the watcher
    thread does that in :meth:`MeetingWatcher.run`). Enumerates
    **all** render endpoints for the target's session and all
    capture endpoints for the mic-claim check. Never raises — a
    failure is reported as ``probe_error`` on the snapshot so the
    watcher can skip the tick instead of misreading it.
    """
    from app.audio.perapp_capture import resolve_target_pid

    try:
        info = resolve_target_pid(process_names)
    except Exception as exc:  # noqa: BLE001 — resolve must not raise
        return SessionSnapshot(probe_error=f"PID lookup failed: {exc}")
    if info is None:
        return SessionSnapshot(process_running=False)
    pid = info["pid"]
    try:
        import comtypes
        from pycaw.api.audiopolicy import (
            IAudioSessionControl2,
            IAudioSessionManager2,
        )
        from pycaw.api.endpointvolume import IAudioMeterInformation
        from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
    except Exception as exc:
        return SessionSnapshot(
            process_running=True,
            probe_error=f"pycaw/comtypes import failed: {exc}")
    try:
        enumerator = comtypes.CoCreateInstance(
            comtypes.GUID(CLSID_MMDEVICE_ENUMERATOR),
            IMMDeviceEnumerator, CLSCTX_ALL)
        state, peak = _find_render_session(
            enumerator, pid, IAudioSessionManager2,
            IAudioMeterInformation, IAudioSessionControl2)
        mic_claimed = _target_has_capture_session(
            enumerator, pid, IAudioSessionManager2,
            IAudioSessionControl2)
    except Exception as exc:  # noqa: BLE001 — probe must not raise
        logger.warning("Auto-detect probe failed: %s", exc)
        return SessionSnapshot(
            process_running=True,
            probe_error=f"session enumeration failed: {exc}")
    return SessionSnapshot(
        process_running=True, session_state=state, peak=peak,
        mic_claimed=mic_claimed)


def _find_render_session(enumerator, pid: int, manager_cls,
                         meter_cls, control2_cls,
                         ) -> tuple[str | None, float | None]:
    """(state, peak) of the target's render session, any endpoint.

    Returns ``(None, None)`` when no session for the pid exists.
    When the pid has sessions on several endpoints, the first
    Active one wins; otherwise the first session found.
    """
    found: tuple[str | None, float | None] | None = None
    for device in _enumerate_endpoints(enumerator, EDATAFLOW_RENDER):
        try:
            manager = _session_manager_for(device, manager_cls)
            session_enum = manager.GetSessionEnumerator()
            # comtypes returns ["out"] params; they are not passed.
            count = session_enum.GetCount()
            if isinstance(count, tuple):
                count = count[0]
            for i in range(count):
                ctl = session_enum.GetSession(i)
                try:
                    # GetSession returns IAudioSessionControl; the pid
                    # lives on IAudioSessionControl2 (pycaw's own
                    # GetAllSessions does the same QueryInterface).
                    ctl = ctl.QueryInterface(control2_cls)
                    proc = ctl.GetProcessId()
                    if isinstance(proc, tuple):
                        proc = proc[0]
                    if proc != pid:
                        continue
                    state, peak = _read_session(ctl, meter_cls)
                    if state == SESSION_STATE_ACTIVE:
                        return state, peak
                    if found is None:
                        found = (state, peak)
                except Exception:
                    continue
        except Exception:
            continue
    return found if found is not None else (None, None)


def _target_has_capture_session(enumerator, pid: int,
                                manager_cls, control2_cls) -> bool:
    """True when the target has an Active session on a capture endpoint."""
    for device in _enumerate_endpoints(enumerator, EDATAFLOW_CAPTURE):
        try:
            manager = _session_manager_for(device, manager_cls)
            session_enum = manager.GetSessionEnumerator()
            # comtypes returns ["out"] params; they are not passed.
            count = session_enum.GetCount()
            if isinstance(count, tuple):
                count = count[0]
            for i in range(count):
                ctl = session_enum.GetSession(i)
                try:
                    # GetSession returns IAudioSessionControl; the pid
                    # lives on IAudioSessionControl2.
                    ctl = ctl.QueryInterface(control2_cls)
                    proc = ctl.GetProcessId()
                    if isinstance(proc, tuple):
                        proc = proc[0]
                    if proc != pid:
                        continue
                    # State only — no meter needed for the mic-claim
                    # vote; GetState is enough.
                    raw_state = ctl.GetState()
                    if isinstance(raw_state, tuple):
                        raw_state = raw_state[0]
                    if raw_state == 1:
                        return True
                except Exception:
                    continue
        except Exception:
            continue
    return False


# --------------------------------------------------------------------------
# Watcher — the state machine (pure logic, COM behind the probe seam)
# --------------------------------------------------------------------------
class MeetingWatcher:
    """Poll the target app's audio sessions; fire meeting events.

    ``on_meeting_started`` / ``on_meeting_ended`` are called on the
    watcher thread — the UI must marshal them onto its own thread
    (see ``app.ui.main_window``'s signal bridge). ``probe`` is the
    test seam: a callable returning a :class:`SessionSnapshot`
    (default: the real Windows probe). ``clock`` is a
    ``time.monotonic``-like callable (tests pass a fake clock).
    ``preroll_buffer`` is an optional
    :class:`app.audio.preroll.PreRollBuffer`: when a meeting starts
    it is drained into :meth:`take_preroll` for the UI layer to
    prepend to the new session. The watcher never writes buffered
    audio anywhere itself.
    """

    def __init__(self, *, process_names: list[str] | None = None,
                 on_meeting_started=None, on_meeting_ended=None,
                 poll_interval: float = AUTO_DETECT_POLL_INTERVAL_SECONDS,
                 sustain_seconds: float = AUTO_DETECT_SUSTAIN_SECONDS,
                 inactive_grace_seconds: float = AUTO_DETECT_INACTIVE_GRACE_SECONDS,
                 silence_fuse_seconds: float = AUTO_DETECT_SILENCE_FUSE_SECONDS,
                 active_mic_seconds: float = AUTO_DETECT_ACTIVE_MIC_SECONDS,
                 peak_threshold: float = AUTO_DETECT_PEAK_THRESHOLD,
                 silence_peak: float = AUTO_DETECT_SILENCE_PEAK,
                 probe=None, clock=None, preroll_buffer=None) -> None:
        self.process_names = list(process_names or TEAMS_PROCESS_NAMES)
        self.on_meeting_started = on_meeting_started
        self.on_meeting_ended = on_meeting_ended
        self.poll_interval = poll_interval
        self.sustain_seconds = sustain_seconds
        self.inactive_grace_seconds = inactive_grace_seconds
        self.silence_fuse_seconds = silence_fuse_seconds
        self.active_mic_seconds = active_mic_seconds
        self.peak_threshold = peak_threshold
        self.silence_peak = silence_peak
        self._probe = probe or self._default_probe
        self._clock = clock or time.monotonic
        self._preroll_buffer = preroll_buffer
        # Pre-roll audio drained at the last meeting start, waiting
        # for the UI layer to take it (take_preroll). {} when there
        # is no buffer or nothing was buffered.
        self._pending_preroll: dict = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # State machine: "idle" (watching for a meeting) or
        # "in_meeting" (a meeting is running; watching for its end).
        self._state = "idle"
        # Manual-stop suppression: set by notify_manual_stop();
        # cleared once the target session goes inactive/absent, so a
        # hand-stopped recording is never immediately re-triggered
        # on the still-active meeting.
        self._suppressed = False
        self._sustained_since: float | None = None
        self._active_since: float | None = None
        # In-meeting stop-gate timers (see _poll_in_meeting):
        # _inactive_since tracks a session gone Inactive/Expired
        # (needs the full 60 s grace before the meeting is over);
        # _fuse_silent_since tracks the 10-min long-silence fuse
        # (Active, no audio, mic unclaimed).
        self._inactive_since: float | None = None
        self._fuse_silent_since: float | None = None

    # ------------------------------------------------------------ state ---
    @property
    def state(self) -> str:
        """``"idle"`` or ``"in_meeting"``."""
        return self._state

    @property
    def is_running(self) -> bool:
        """True while the watcher thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def _default_probe(self) -> SessionSnapshot:
        return _probe_windows(self.process_names)

    # ---------------------------------------------------------- control ---
    def start(self) -> None:
        """Start polling on a background thread.

        Raises RuntimeError when auto-detect is unavailable on this
        machine (non-Windows / pycaw missing) — the caller reports
        the reason instead of starting a watcher that can never fire.
        """
        available, reason = auto_detect_available()
        if not available:
            raise RuntimeError(
                f"Meeting auto-detect unavailable: {reason}")
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="offair-auto-detect", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling (blocks briefly for the thread to exit)."""
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.poll_interval * 2))

    def notify_manual_stop(self) -> None:
        """The user stopped a recording by hand.

        Suppresses auto-restart until the target session goes
        inactive/absent — otherwise the still-active meeting would
        re-trigger a recording the user just stopped. Also resets
        the in-meeting state: the watcher goes back to watching.
        """
        with self._lock:
            self._suppressed = True
            self._state = "idle"
            self._reset_counters()

    # ------------------------------------------------------------- poll ---
    def poll_once(self) -> None:
        """One poll tick: read the probe, run the gates, fire events.

        Public so tests can drive the state machine deterministically
        (the thread loop just calls this every ``poll_interval``).
        """
        try:
            snapshot = self._probe()
        except Exception as exc:  # noqa: BLE001 — a bad probe is a skip
            logger.warning("Auto-detect probe raised: %s", exc)
            return
        if snapshot.probe_error is not None:
            # Enumeration itself failed — a COM hiccup, not a meeting
            # ending. Skip the tick; keep state and counters.
            logger.debug("Auto-detect probe error, skipping tick: %s",
                         snapshot.probe_error)
            return
        with self._lock:
            if self._state == "in_meeting":
                self._poll_in_meeting(snapshot)
            else:
                self._poll_idle(snapshot)

    def _poll_idle(self, snap: SessionSnapshot) -> None:
        now = self._clock()
        if self._suppressed:
            # Re-arm only once the session is gone/inactive — the
            # meeting the user stopped has actually ended.
            if (not snap.process_running
                    or snap.session_state in (None, SESSION_STATE_INACTIVE,
                                              SESSION_STATE_EXPIRED)):
                self._suppressed = False
                self._reset_counters()
            return
        if (not snap.process_running
                or snap.session_state != SESSION_STATE_ACTIVE):
            self._reset_counters()
            return
        # Session is Active — track how long it has been so, and
        # how long its peak has been above the threshold.
        if self._active_since is None:
            self._active_since = now
        peak = snap.peak
        if peak is not None and peak >= self.peak_threshold:
            if self._sustained_since is None:
                self._sustained_since = now
            if now - self._sustained_since >= self.sustain_seconds:
                self._fire_started()
                return
        else:
            self._sustained_since = None
        # Secondary rule: Active for >30 s with the mic claimed is
        # a meeting even with no peak (all-muted, nobody speaking).
        if (snap.mic_claimed and self._active_since is not None
                and now - self._active_since >= self.active_mic_seconds):
            self._fire_started()

    def _poll_in_meeting(self, snap: SessionSnapshot) -> None:
        """Stop gate (revised Nov 8 2026): silence mid-meeting NEVER
        stops a recording. Stop only on:

        (b) the target process exiting — immediate;
        (a) the session going Inactive/Expired (or disappearing)
            and STAYING that way for ``inactive_grace_seconds``
            (60 s) — a single inactive poll never stops anything,
            and the session coming back Active cancels the grace
            (transient renegotiation mid-call);
        (c) the long-silence fuse — session Active with peak at or
            below ``silence_peak`` AND the mic unclaimed,
            continuously for ``silence_fuse_seconds`` (10 min).
            A stuck/leftover session, not a real pause. Any audio,
            or the mic being claimed, resets the fuse.
        """
        now = self._clock()
        if not snap.process_running:
            self._fire_ended()
            return
        if snap.session_state in (None, SESSION_STATE_INACTIVE,
                                  SESSION_STATE_EXPIRED):
            if self._inactive_since is None:
                self._inactive_since = now
            if now - self._inactive_since >= self.inactive_grace_seconds:
                self._fire_ended()
            return
        # Session is Active again — cancel any pending grace.
        self._inactive_since = None
        peak = snap.peak
        if (peak is not None and peak <= self.silence_peak
                and not snap.mic_claimed):
            if self._fuse_silent_since is None:
                self._fuse_silent_since = now
            if now - self._fuse_silent_since >= self.silence_fuse_seconds:
                self._fire_ended()
                return
        else:
            self._fuse_silent_since = None

    # ------------------------------------------------------------ events --
    def take_preroll(self) -> dict:
        """Take the pre-roll drained at the last meeting start.

        Returns ``{"mic": samples, "system": samples}`` (int16 mono
        arrays) — the shape :meth:`app.audio.capture.DualCapture.start`
        accepts as ``preroll_samples`` — or ``{}`` when no buffer is
        attached or nothing was buffered. One-shot: a second call
        returns ``{}``. The UI layer calls this in its
        meeting-started handler, before starting capture; if it is
        never called, the drained audio is simply discarded (it was
        memory-only and is never written anywhere by this watcher).
        """
        with self._lock:
            pending, self._pending_preroll = self._pending_preroll, {}
        return pending

    def _fire_started(self) -> None:
        self._state = "in_meeting"
        self._reset_counters()
        # Drain the pre-roll buffer now — at the start gate, not
        # later: audio buffered after this moment belongs to the
        # recording itself, not to its lead-in. Draining also
        # clears the buffer, so a second meeting cannot inherit
        # the first meeting's lead-in.
        if self._preroll_buffer is not None:
            try:
                self._pending_preroll = self._preroll_buffer.drain_all()
            except Exception:  # noqa: BLE001 — a bad buffer is logged
                logger.exception("Pre-roll drain failed; starting "
                                 "without pre-roll")
                self._pending_preroll = {}
        logger.info("Auto-detect: meeting started (Teams session active "
                    "with sustained audio)")
        if self.on_meeting_started is not None:
            try:
                self.on_meeting_started()
            except Exception:  # noqa: BLE001 — a bad callback is logged
                logger.exception("on_meeting_started callback failed")

    def _fire_ended(self) -> None:
        self._state = "idle"
        self._reset_counters()
        logger.info("Auto-detect: meeting ended")
        if self.on_meeting_ended is not None:
            try:
                self.on_meeting_ended()
            except Exception:  # noqa: BLE001 — a bad callback is logged
                logger.exception("on_meeting_ended callback failed")

    def _reset_counters(self) -> None:
        self._sustained_since = None
        self._active_since = None
        self._inactive_since = None
        self._fuse_silent_since = None

    # ------------------------------------------------------------ thread --
    def _run(self) -> None:
        """Poll loop: COM is initialised on this thread, then 1 Hz."""
        com_initialised = False
        if sys.platform == "win32":
            try:
                # Import comtypes FIRST: it STA-initialises this thread
                # at import time, and a later import on an
                # already-MTA thread raises RPC_E_CHANGED_MODE ("Cannot
                # change thread mode after it is set") — which is
                # exactly what the probe's own `import comtypes` would
                # hit. With comtypes imported, coinit_mta() returns
                # RPC_E_CHANGED_MODE and the thread stays STA, which
                # is fine for the probe's synchronous pycaw calls.
                import comtypes  # noqa: F401
                # Same ole32-direct call as perapp_capture.coinit_mta:
                # the comtypes wrapper takes one argument in 1.4.17
                # and raises RPC_E_CHANGED_MODE instead of returning
                # it (it STA-initialises the importing thread).
                from app.audio.perapp_capture import coinit_mta

                hr = coinit_mta()
                if hr == 0:
                    com_initialised = True
                elif (hr & 0xFFFFFFFF) == RPC_E_CHANGED_MODE:
                    logger.warning(
                        "Auto-detect: COM already initialised on the "
                        "watcher thread — continuing")
            except Exception:
                logger.exception(
                    "Auto-detect: CoInitializeEx failed; the probe "
                    "will report errors and ticks will be skipped")
        try:
            while not self._stop_event.is_set():
                self.poll_once()
                self._stop_event.wait(timeout=self.poll_interval)
        finally:
            if com_initialised:
                try:
                    import comtypes
                    comtypes.CoUninitialize()
                except Exception:
                    pass


__all__ = [
    "AUTO_DETECT_POLL_INTERVAL_SECONDS",
    "AUTO_DETECT_SUSTAIN_SECONDS",
    "AUTO_DETECT_INACTIVE_GRACE_SECONDS",
    "AUTO_DETECT_SILENCE_FUSE_SECONDS",
    "AUTO_DETECT_ACTIVE_MIC_SECONDS",
    "AUTO_DETECT_PEAK_THRESHOLD",
    "AUTO_DETECT_SILENCE_PEAK",
    "MeetingWatcher",
    "SessionSnapshot",
    "auto_detect_available",
]
