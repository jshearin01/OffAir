"""True per-app audio capture for Windows (WASAPI process loopback).

Replaces the honest-but-limited :class:`app.audio.capture.PerAppCapture`,
which records system-wide WASAPI loopback and says so. This module
captures **only the target process's audio** (e.g. Teams) using the
Windows Application Loopback API.

Mechanism (verified from primary sources — see ``docs/PER_APP_CAPTURE.md``
for the full write-up with links):

* ``ActivateAudioInterfaceAsync`` on the virtual device
  ``VAD\\Process_Loopback`` with ``AUDIOCLIENT_ACTIVATION_PARAMS`` /
  ``AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS`` (``TargetProcessId`` +
  ``PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE``).
  Microsoft Learn, "Application loopback audio capture" sample:
  requires **Windows 10 build 20348 or later**; if the target
  process has no rendering streams the capture yields silence.
  OBS 28+ uses exactly this for "Application Audio Capture"
  (``plugins/win-wasapi/win-wasapi.cpp``); GStreamer ``wasapi2src``
  and Chromium's low-latency input path ship it too.
* The completion handler **must also implement ``IAgileObject``**
  (marker interface, no methods) or activation fails with
  ``E_ILLEGAL_METHOD_CALL`` (0x8000000E) and nothing says why —
  verified in the Voxis dev.to write-up, the GStreamer
  ``Wasapi2ActivationHandler`` (answers ``QueryInterface`` for
  ``IUnknown`` / ``IAgileObject`` /
  ``IActivateAudioInterfaceCompletionHandler``), the Sunshine/Hydra
  issue #844 probe, and Microsoft Learn
  (``IActivateAudioInterfaceCompletionHandler``: "The implementation
  must be agile").
* Pure Python is possible: ``comtypes`` + ``pycaw``. pycaw
  (20260927, verified by inspecting the wheel) defines the
  battle-tested ``IAudioClient`` vtbl, ``WAVEFORMATEX``,
  ``IAudioSessionManager2`` / ``IAudioSessionControl2`` /
  ``IAudioMeterInformation`` and ``AudioUtilities.GetAllSessions()``
  — but it does **not** define ``IAudioCaptureClient`` or the
  activation interfaces, so those are hand-defined here from the
  Windows SDK headers (IIDs cross-checked against the ``winapi``
  crate, which is generated from the SDK, and the ReactOS/Wine
  IDL mirrors):
  ``IActivateAudioInterfaceCompletionHandler``
  ``41D949AB-9862-444A-80F6-C261334DA5EB``,
  ``IActivateAudioInterfaceAsyncOperation``
  ``72A22D78-CDE4-431D-B8CC-843A71199B6D``,
  ``IAgileObject`` ``94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90``,
  ``IAudioCaptureClient`` ``C8ADBD64-E71E-48A0-A4DE-185C395CD317``
  (``GetBuffer`` takes **5** parameters — a 4-parameter definition
  causes access violations; see the XDA anecdote in
  ``docs/PER_APP_CAPTURE.md``).
* Format: the client is initialised with the requested format
  (48 kHz mono int16) directly — Voxis reports this works — and
  falls back to ``GetMixFormat`` + conversion (float/stereo/rate)
  on ``AUDCLNT_E_UNSUPPORTED_FORMAT``. ``GetDevicePeriod`` returns
  ``E_NOTIMPL`` on process loopback (GStreamer note), so the read
  loop polls ``GetNextPacketSize`` instead of using an event handle.
  A ``SILENT`` buffer flag means the packet is digital silence.

Failure modes (all raise :class:`PerAppCaptureError`, synchronously
from ``start()`` — never a silent dead track):

* not Windows / build < 20348 / ``pycaw`` missing → unavailable
  (``per_app_capture_available()``); ``DualCapture`` falls back to
  system-wide loopback in ``"auto"`` mode.
* target process not running → no PID to capture (same fallback).
* activation failure (incl. the ``E_ILLEGAL_METHOD_CALL`` trap) →
  error naming the stage and HRESULT.
* target running but not rendering yet → activation succeeds and
  the stream yields silence until it does; ``describe_target_session``
  reports that state so the UI can warn instead of looking broken.
* the Teams desktop client (``ms-teams.exe``) is the known
  exception: process loopback returns silence for it even mid-call
  with active sessions (Microsoft windows-classic-samples#414;
  Nov 10 + Nov 24 2026 on the work PC) — ``"auto"`` mode records
  it system-wide instead; see :data:`TEAMS_DESKTOP_EXES`. The
  activation target is the session's pid when one is known
  (:func:`resolve_capture_pid`), not blindly the tree root.

The microphone side is untouched: the mic track was always
per-user. Only the system track changes.

What is **not** verified here: no real WASAPI run exists yet — this
Linux VM has no Windows audio stack, so the COM path is exercised
only through injected fake streams (see ``tests/test_perapp_capture.py``).
The first real Windows run is the open gate; see
``docs/PER_APP_CAPTURE.md`` for the exact test procedure.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.audio.capture import WASAPICapture, _to_int16_mono

logger = logging.getLogger(__name__)

#: Virtual device path for process-loopback activation
#: (``VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`` in
#: ``audioclientactivationparams.h``).
VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"

#: Minimum Windows build for process loopback (Microsoft sample note).
MIN_WINDOWS_BUILD_PER_APP = 20348

#: Mode strings recorded in ``metadata.json``.
MODE_PER_APP = "per-app"
MODE_LOOPBACK_FALLBACK = "system-loopback-fallback"

#: ``AUDIOCLIENT_ACTIVATION_TYPE`` values (audioclientactivationparams.h).
ACTIVATION_TYPE_DEFAULT = 0
ACTIVATION_TYPE_PROCESS_LOOPBACK = 1

#: ``PROCESS_LOOPBACK_MODE`` values.
LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE = 1

#: ``AUDCLNT_SHAREMODE_SHARED`` / ``AUDCLNT_STREAMFLAGS_LOOPBACK``
#: (audioclient.h: ``AUDCLNT_STREAMFLAGS_CROSSPROCESS`` 0x00010000,
#: ``LOOPBACK`` 0x00020000, ``EVENTCALLBACK`` 0x00040000). Note
#: 0x00000002 is ``AUDCLNT_BUFFERFLAGS_SILENT`` — a *GetBuffer* flag,
#: not a stream flag: passing it to ``IAudioClient::Initialize``
#: fails on real Windows with ``AUDCLNT_E_INVALID_STREAM_FLAG``
#: (0x88890021). Found Nov 8 2026 in the Windows VM (the probe
#: failed until this value was corrected).
AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000

#: ``IAudioCaptureClient::GetBuffer`` flag: packet is digital silence.
AUDCLNT_BUFFERFLAGS_SILENT = 0x00000002
#: Same interface: discontinuity in the captured data (log, don't fail).
AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY = 0x00000001

#: ``AUDCLNT_E_UNSUPPORTED_FORMAT`` (audioclient.h).
AUDCLNT_E_UNSUPPORTED_FORMAT = 0x88890008
#: ``E_ILLEGAL_METHOD_CALL`` — the completion-handler-isn't-agile trap.
E_ILLEGAL_METHOD_CALL = 0x8000000E
#: ``RPC_E_CHANGED_MODE`` — CoInitializeEx on an already-initialised
#: thread; safe to continue (the handler is agile, so the apartment
#: the activation lands in does not matter).
RPC_E_CHANGED_MODE = 0x80010106

#: ``WAVE_FORMAT_PCM`` / ``WAVE_FORMAT_IEEE_FLOAT``.
WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3

#: ``VT_BLOB`` (wtypes.h) — the PROPVARIANT type carrying the
#: activation parameters.
VT_BLOB = 0x0041

#: ``AudioSessionState`` (audiopolicy.h): 0 inactive, 1 active, 2 expired.
AUDIO_SESSION_STATE_ACTIVE = 1

#: IIDs, cross-checked against the ``winapi`` crate (generated from the
#: Windows SDK headers) and the ReactOS/Wine IDL mirrors — see the
#: module docstring. These are Microsoft-defined constants, not
#: invented values.
IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER = (
    "{41D949AB-9862-444A-80F6-C261334DA5EB}")
IID_IACTIVATE_AUDIO_INTERFACE_ASYNC_OPERATION = (
    "{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
IID_IAGILE_OBJECT = "{94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90}"
IID_IAUDIO_CAPTURE_CLIENT = "{C8ADBD64-E71E-48A0-A4DE-185C395CD317}"

#: Default activation wait (the async call normally completes in ms).
ACTIVATION_TIMEOUT_SECONDS = 10.0
#: WASAPI ring-buffer duration requested at Initialize (100-ns units;
#: 2,000,000 = 200 ms, as used by Voxis).
BUFFER_DURATION_100NS = 2_000_000


class PerAppCaptureError(RuntimeError):
    """Per-app capture failed (availability, activation, or stream)."""


# --------------------------------------------------------------------------
# Activation parameter structs (pure ctypes — importable/testable anywhere)
# --------------------------------------------------------------------------
class _ProcessLoopbackParams(ctypes.Structure):
    """``AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS``.

    ``DWORD TargetProcessId; PROCESS_LOOPBACK_MODE ProcessLoopbackMode;``
    (``PROCESS_LOOPBACK_MODE`` is a C enum → ``c_int``.)
    """

    _fields_ = [
        ("TargetProcessId", ctypes.c_uint32),
        ("ProcessLoopbackMode", ctypes.c_int),
    ]


class _ActivationParamsUnion(ctypes.Union):
    _fields_ = [
        ("ProcessLoopbackParams", _ProcessLoopbackParams),
    ]


class _ActivationParams(ctypes.Structure):
    """``AUDIOCLIENT_ACTIVATION_PARAMS`` (loopback member only).

    The real header's union has other members for other activation
    types, but for ``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK``
    only the leading fields are read, so this 12-byte layout
    (``ActivationType`` + ``ProcessLoopbackParams``) is what the OS
    consumes — matching the Microsoft sample and the Voxis/GStreamer
    copies.
    """

    _fields_ = [
        ("ActivationType", ctypes.c_int),
        ("u", _ActivationParamsUnion),
    ]


class _PropVariantBlob(ctypes.Structure):
    """A ``PROPVARIANT`` carrying a ``VT_BLOB``.

    Layout: ``vt`` + 3 reserved words (8 bytes), then the ``BLOB``
    (``ULONG cbSize`` at offset 8, ``BYTE *pBlobData`` at offset 16
    on 64-bit) — 24 bytes total. Only the blob member is populated;
    the rest of the PROPVARIANT union is never read for ``VT_BLOB``.
    """

    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("cbSize", ctypes.c_ulong),
        ("pBlobData", ctypes.c_void_p),
    ]


def build_activation_params(pid: int, mode: int = LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
                            ) -> tuple[_ActivationParams, _PropVariantBlob]:
    """Pack the activation parameters for ``pid`` into a PROPVARIANT blob.

    Returns ``(params, propvariant)`` — keep both alive until
    ``ActivateAudioInterfaceAsync`` returns; the PROPVARIANT's blob
    pointer borrows ``params``.
    """
    params = _ActivationParams()
    params.ActivationType = ACTIVATION_TYPE_PROCESS_LOOPBACK
    params.u.ProcessLoopbackParams.TargetProcessId = pid
    params.u.ProcessLoopbackParams.ProcessLoopbackMode = mode
    pv = _PropVariantBlob()
    pv.vt = VT_BLOB
    pv.cbSize = ctypes.sizeof(params)
    pv.pBlobData = ctypes.cast(
        ctypes.byref(params), ctypes.c_void_p).value
    return params, pv


# --------------------------------------------------------------------------
# Availability + target resolution (no COM — safe anywhere)
# --------------------------------------------------------------------------
def per_app_capture_available() -> tuple[bool, str | None]:
    """Is WASAPI process-loopback capture usable on this machine?

    Returns ``(True, None)`` or ``(False, reason)``. Never raises.
    """
    if sys.platform != "win32":
        return False, ("Per-app capture needs Windows (WASAPI process "
                       f"loopback); this machine is {sys.platform}.")
    try:
        build = sys.getwindowsversion().build
    except Exception as exc:  # pragma: no cover - Windows-only branch
        return False, f"Could not determine the Windows build: {exc}"
    if build < MIN_WINDOWS_BUILD_PER_APP:
        return False, (
            "Per-app capture needs Windows 10 build "
            f"{MIN_WINDOWS_BUILD_PER_APP} or later (this machine: build "
            f"{build}). System-wide loopback is used instead.")
    try:
        import pycaw  # noqa: F401  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - Windows-only branch
        return False, (
            "Per-app capture needs the 'pycaw' package installed "
            f"({exc}); install it with `pip install -r requirements.txt`.")
    return True, None


def resolve_target_pid(process_names: list[str],
                       process_iter=None) -> dict | None:
    """Resolve the target process to capture.

    Returns ``{"pid": int, "name": str}`` or ``None`` when no target
    process is running. When several processes match (Teams runs
    helper processes under the same exe name), the **tree root** is
    preferred — the process whose parent is not also a target process
    — because ``INCLUDE_TARGET_PROCESS_TREE`` captures the root and
    all its children. ``process_iter`` is a test seam (defaults to
    ``psutil.process_iter``); each yielded process must expose
    ``.info`` with ``pid`` / ``name`` / ``ppid``.
    """
    import psutil

    iterator = process_iter if process_iter is not None else psutil.process_iter
    wanted = {name.lower() for name in process_names}
    matches: list[dict] = []
    for proc in iterator(["pid", "name", "ppid"]):
        try:
            info = proc.info
        except Exception:
            continue
        name = info.get("name") or ""
        if name.lower() in wanted:
            matches.append({"pid": info["pid"], "name": name,
                            "ppid": info.get("ppid")})
    if not matches:
        return None
    pids = {m["pid"] for m in matches}
    roots = [m for m in matches if m.get("ppid") not in pids]
    chosen = roots[0] if roots else matches[0]
    return {"pid": chosen["pid"], "name": chosen["name"]}


def resolve_capture_pid(process_names: list[str], *,
                        sessions=None, process_iter=None) -> dict | None:
    """PID to activate process loopback for: the pid actually rendering.

    Prefers the pid of the app's audio session (direct exe match,
    else a session in the app's process tree — see
    :func:`app.audio.app_sessions.find_session_for_app`), because
    activating the tree root can target a tree that holds no audio:
    Teams runs several ``ms-teams.exe`` trees and the root
    heuristic picks ``roots[0]``. Falls back to
    :func:`resolve_target_pid` (tree root) when no session is
    known. ``sessions`` is an explicit
    :func:`app.audio.app_sessions.list_audio_sessions` result for
    tests; ``None`` lists live (``[]`` off Windows / on failure —
    never raises for a listing failure).
    """
    if sessions is None:
        try:
            from app.audio.app_sessions import list_audio_sessions

            sessions = list_audio_sessions()
        except Exception:
            sessions = []
    from app.audio.app_sessions import find_session_for_app

    for name in process_names:
        info = find_session_for_app(sessions, name,
                                    process_iter=process_iter)
        if info is not None:
            return {"pid": info.pid, "name": info.exe_name}
    return resolve_target_pid(process_names, process_iter)


#: Exes of the Teams desktop client. WASAPI process loopback
#: returns silence for it even mid-call with active sessions
#: (Microsoft windows-classic-samples#414, reproduced on several
#: machines/versions; Nov 10 + Nov 24 2026 on the work PC), while
#: device loopback captures it fine — so ``"auto"`` mode records
#: Teams desktop meetings system-wide instead of trusting per-app
#: capture. Explicit ``"per-app"`` mode still tries per-app: the
#: user chose it, and a loud activation failure stays loud.
TEAMS_DESKTOP_EXES = frozenset({"ms-teams.exe", "teams.exe"})


def select_system_capture_class(capture_system_mode: str,
                                process_name: str | None = None):
    """Which system-capture class :class:`DualCapture` should use.

    ``"per-app"`` → :class:`ProcessLoopbackCapture` (fails loudly if
    it cannot start); ``"loopback"`` → the old
    :class:`app.audio.capture.PerAppCapture` (system-wide, always
    works); ``"auto"`` (default) → per-app when available, else the
    loopback class — except the Teams desktop client (see
    :data:`TEAMS_DESKTOP_EXES`), which ``"auto"`` records
    system-wide: process loopback returns silence for it. Never
    raises; unknown modes fall back to ``"auto"``.
    """
    from app.audio.capture import PerAppCapture

    if capture_system_mode == "loopback":
        return PerAppCapture
    if capture_system_mode == "per-app":
        return ProcessLoopbackCapture
    if (process_name or "").lower() in TEAMS_DESKTOP_EXES:
        return PerAppCapture
    available, _ = per_app_capture_available()
    return ProcessLoopbackCapture if available else PerAppCapture


def describe_target_session(pid: int) -> dict | None:
    """Best-effort probe: is ``pid`` currently rendering audio?

    Returns ``{"pid", "active", "peak", "display_name"}`` or ``None``
    when the probe cannot run (non-Windows, pycaw missing, COM not
    initialised on this thread, no session for the pid). Never raises.
    ``active`` is the WASAPI session state; ``peak`` is the session's
    current peak level (0.0–1.0, ``None`` when unreadable) — a session
    can be active yet silent (Teams open, no call), which is the case
    the UI warns about. Enumerates the **default render** device's
    sessions; Teams rendering to a non-default device is missed
    (documented limitation).
    """
    if sys.platform != "win32":
        return None
    try:
        from pycaw.api.endpointvolume import IAudioMeterInformation
        from pycaw.utils import AudioUtilities

        for session in AudioUtilities.GetAllSessions():
            try:
                if session.ProcessId != pid:
                    continue
                active = session.State == AUDIO_SESSION_STATE_ACTIVE
                peak: float | None = None
                try:
                    meter = session._ctl.QueryInterface(
                        IAudioMeterInformation)
                    # comtypes returns ["out"] params; they are not
                    # passed — the byref form raised TypeError and
                    # peak silently stayed None on real Windows.
                    res = meter.GetPeakValue()
                    if isinstance(res, tuple):
                        res = res[0]
                    peak = float(res)
                except Exception:
                    peak = None
                name = None
                try:
                    name = session.DisplayName or None
                except Exception:
                    name = None
                return {"pid": pid, "active": active, "peak": peak,
                        "display_name": name}
            except Exception:
                continue
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# Format conversion (pure numpy — testable anywhere)
# --------------------------------------------------------------------------
def _resample_linear(samples: np.ndarray, src_rate: int,
                     dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample of int16 mono samples.

    Speech-grade, not studio-grade — good enough for a meeting track
    when the mix format's rate differs from the requested 48 kHz.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    n_out = int(round(len(samples) * dst_rate / src_rate))
    if n_out <= 0:
        return samples[:0]
    x_old = np.arange(len(samples))
    x_new = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x_new, x_old,
                     samples.astype(np.float64)).astype(np.int16)


def _convert_mix_chunk(data: bytes, *, rate: int, channels: int,
                       format_tag: int, bits: int,
                       target_rate: int) -> bytes:
    """Convert one raw mix-format packet to 48 kHz mono int16 bytes.

    Handles the mix formats WASAPI can return: PCM int16/int32 and
    IEEE-float float32/float64, mono or stereo. Anything else raises
    :class:`PerAppCaptureError` (fail loudly rather than write garbage).
    """
    if format_tag == WAVE_FORMAT_IEEE_FLOAT and bits == 32:
        arr = np.frombuffer(data, dtype=np.float32)
    elif format_tag == WAVE_FORMAT_IEEE_FLOAT and bits == 64:
        arr = np.frombuffer(data, dtype=np.float64)
    elif format_tag == WAVE_FORMAT_PCM and bits == 16:
        arr = np.frombuffer(data, dtype=np.int16)
    elif format_tag == WAVE_FORMAT_PCM and bits == 32:
        arr = (np.frombuffer(data, dtype=np.int32) >> 16).astype(np.int16)
    else:
        raise PerAppCaptureError(
            f"Unsupported mix format: tag={format_tag} bits={bits} "
            f"channels={channels} rate={rate} — cannot convert to "
            "48 kHz mono int16.")
    mono = _to_int16_mono(arr, channels)
    if rate != target_rate:
        mono = _resample_linear(mono, rate, target_rate)
    return mono.tobytes()


# --------------------------------------------------------------------------
# COM layer (Windows-only; built lazily so the module imports on Linux)
# --------------------------------------------------------------------------
_COM_LAYER: SimpleNamespace | None = None


def _com_layer() -> SimpleNamespace:
    """Build (once) the comtypes interface classes for the COM path.

    Raises :class:`PerAppCaptureError` on non-Windows or when
    comtypes/pycaw cannot be imported. The interface definitions
    below are hand-written from the SDK headers — pycaw does not
    ship ``IAudioCaptureClient`` or the activation interfaces.
    """
    global _COM_LAYER
    if _COM_LAYER is not None:
        return _COM_LAYER
    if sys.platform != "win32":
        raise PerAppCaptureError(
            "Per-app capture is Windows-only "
            f"(this machine is {sys.platform}).")
    try:
        import comtypes
        from comtypes import COMMETHOD, COMObject, GUID, HRESULT, IUnknown, POINTER
        from ctypes import c_ubyte as BYTE
        from ctypes import c_uint32 as UINT32
        from ctypes import c_uint64 as UINT64
        from ctypes import c_ulong as DWORD
        from pycaw.api.audioclient import IAudioClient, WAVEFORMATEX
    except Exception as exc:
        raise PerAppCaptureError(
            "Per-app capture needs comtypes + pycaw "
            f"(`pip install -r requirements.txt`): {exc}") from exc

    class IAgileObject(IUnknown):
        """Marker interface — no methods. See module docstring."""

        _iid_ = GUID(IID_IAGILE_OBJECT)
        _methods_ = ()

    class IActivateAudioInterfaceAsyncOperation(IUnknown):
        _iid_ = GUID(IID_IACTIVATE_AUDIO_INTERFACE_ASYNC_OPERATION)
        _methods_ = (
            COMMETHOD([], HRESULT, "GetActivateResult",
                      (["out"], POINTER(HRESULT), "activateResult"),
                      (["out"], POINTER(POINTER(IUnknown)),
                       "activatedInterface")),
        )

    class IActivateAudioInterfaceCompletionHandler(IUnknown):
        _iid_ = GUID(IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER)
        _methods_ = (
            COMMETHOD([], HRESULT, "ActivateCompleted",
                      (["in"],
                       POINTER(IActivateAudioInterfaceAsyncOperation),
                       "activateOperation")),
        )

    class IAudioCaptureClient(IUnknown):
        # GetBuffer takes FIVE parameters (ppData, pNumFramesToRead,
        # pdwFlags, pu64DevicePosition, pu64QPCPosition) — a
        # four-parameter definition causes access violations.
        _iid_ = GUID(IID_IAUDIO_CAPTURE_CLIENT)
        _methods_ = (
            COMMETHOD([], HRESULT, "GetBuffer",
                      (["out"], POINTER(POINTER(BYTE)), "ppData"),
                      (["out"], POINTER(UINT32), "pNumFramesToRead"),
                      (["out"], POINTER(DWORD), "pdwFlags"),
                      (["out"], POINTER(UINT64), "pu64DevicePosition"),
                      (["out"], POINTER(UINT64), "pu64QPCPosition")),
            COMMETHOD([], HRESULT, "ReleaseBuffer",
                      (["in"], UINT32, "NumFramesRead")),
            COMMETHOD([], HRESULT, "GetNextPacketSize",
                      (["out"], POINTER(UINT32),
                       "pNumFramesInNextPacket")),
        )

    class _ActivationHandler(COMObject):
        """Completion handler for ActivateAudioInterfaceAsync.

        Implements ``IActivateAudioInterfaceCompletionHandler`` **and**
        ``IAgileObject`` — without the latter, activation returns
        ``E_ILLEGAL_METHOD_CALL`` (see module docstring). Windows
        calls ``ActivateCompleted`` on an MTA worker thread; it
        records the result and sets the event the starter waits on.
        """

        _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler,
                            IAgileObject]

        def __init__(self) -> None:
            super().__init__()
            self.event = threading.Event()
            self.activate_hr: int | None = None
            self.activated = None

        def ActivateCompleted(self, activateOperation):  # noqa: N802 - COM name
            try:
                # comtypes returns ["out"] params; they are not passed:
                # calling with byref args raises "TypeError: call takes
                # exactly 1 arguments". GetActivateResult() returns
                # (activateResult, activatedInterface); some comtypes
                # versions prepend the method HRESULT (3-tuple). The
                # 2-tuple shape is what real Windows returned Nov 8 2026.
                res = activateOperation.GetActivateResult()
                if isinstance(res, tuple) and len(res) == 2:
                    hr_value, iface = res
                    method_hr = 0
                elif isinstance(res, tuple) and len(res) == 3:
                    method_hr, hr_value, iface = res
                else:
                    raise TypeError(
                        f"unexpected GetActivateResult shape: {res!r}")
                # GetActivateResult returns S_OK itself and reports the
                # activation outcome through *activateResult.
                self.activate_hr = hr_value if method_hr == 0 else method_hr
                self.activated = iface
            except Exception:
                logger.exception("ActivateCompleted failed")
                self.activate_hr = E_ILLEGAL_METHOD_CALL
                self.activated = None
            finally:
                self.event.set()
            return 0  # S_OK

    _mmdevapi = ctypes.windll.mmdevapi
    _mmdevapi.ActivateAudioInterfaceAsync.argtypes = [
        ctypes.c_wchar_p,  # deviceInterfacePath
        ctypes.POINTER(GUID),  # riid (IAudioClient)
        ctypes.POINTER(_PropVariantBlob),  # activationParams
        ctypes.POINTER(IActivateAudioInterfaceCompletionHandler),
        ctypes.POINTER(
            ctypes.POINTER(IActivateAudioInterfaceAsyncOperation)),
    ]
    _mmdevapi.ActivateAudioInterfaceAsync.restype = ctypes.c_long  # HRESULT

    _COM_LAYER = SimpleNamespace(
        comtypes=comtypes,
        IAudioClient=IAudioClient,
        WAVEFORMATEX=WAVEFORMATEX,
        IAudioCaptureClient=IAudioCaptureClient,
        IActivateAudioInterfaceAsyncOperation=(
            IActivateAudioInterfaceAsyncOperation),
        IActivateAudioInterfaceCompletionHandler=(
            IActivateAudioInterfaceCompletionHandler),
        IAgileObject=IAgileObject,
        ActivationHandler=_ActivationHandler,
        mmdevapi=_mmdevapi,
    )
    return _COM_LAYER


def _check_hr(hr: int, stage: str) -> None:
    """Raise PerAppCaptureError for a FAILED HRESULT (negative as int32)."""
    if hr is None:
        return
    if hr & 0x80000000:
        hint = ""
        if (hr & 0xFFFFFFFF) == E_ILLEGAL_METHOD_CALL:
            hint = (" (E_ILLEGAL_METHOD_CALL — the completion handler "
                    "was not recognised as agile; see docs/PER_APP_CAPTURE.md)")
        raise PerAppCaptureError(
            f"{stage} failed: HRESULT 0x{hr & 0xFFFFFFFF:08X}{hint}")


#: ``COINIT_MULTITHREADED`` (0x0) — the apartment flag for
#: :func:`coinit_mta`.
COINIT_MULTITHREADED = 0x0


def coinit_mta() -> int:
    """Initialise COM on this thread (MTA) and return the HRESULT.

    Calls ``ole32.CoInitializeEx`` directly via ctypes (``restype``
    ``c_long``) instead of the comtypes wrapper, for two reasons
    verified on real Windows Nov 8 2026 (comtypes 1.4.17):

    * the wrapper exposes ``CoInitializeEx(flags=None)`` — a single
      optional argument — so the two-argument call raises
      ``TypeError`` before any COM work;
    * ``import comtypes`` STA-initialises the importing thread, so a
      wrapper call would *raise* ``RPC_E_CHANGED_MODE`` instead of
      returning it, and the caller's already-initialised branch could
      never run.

    The direct call returns the HRESULT as a value: 0 (``S_OK``) means
    this thread now owns the COM init and must call
    ``CoUninitialize``; ``RPC_E_CHANGED_MODE`` means the thread was
    already initialised in another apartment (safe to continue — the
    activation handler is agile). Any other value is a real failure
    for the caller to report.
    """
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    return ole32.CoInitializeEx(None, COINIT_MULTITHREADED)


class _ComProcessLoopbackStream:
    """The real WASAPI process-loopback stream (Windows only).

    Lifecycle: ``start()`` activates the audio client for ``pid``,
    initialises it (requested format, else the mix format),
    obtains the capture client and starts the stream. ``read_chunk()``
    polls ``GetNextPacketSize`` and returns one packet as 48 kHz mono
    int16 bytes (converting when the mix format differs), or ``None``
    on timeout. ``stop()`` stops the client; ``close()`` releases the
    COM references and uninitialises COM on this thread.
    """

    def __init__(self, pid: int, sample_rate: int = 48000,
                 channels: int = 1, *,
                 activation_timeout: float = ACTIVATION_TIMEOUT_SECONDS,
                 buffer_duration_100ns: int = BUFFER_DURATION_100NS) -> None:
        self.pid = pid
        self.sample_rate = sample_rate
        self.channels = channels
        self.activation_timeout = activation_timeout
        self.buffer_duration_100ns = buffer_duration_100ns
        self._com: SimpleNamespace | None = None
        self._handler = None
        self._async_op = None
        self._client = None
        self._capture = None
        self._mix_format: dict | None = None
        self._block_align = channels * 2
        self._com_initialised = False
        self._started = False

    # ------------------------------------------------------------ start ---
    def start(self) -> None:
        com = _com_layer()
        self._com = com
        self._coinit()
        try:
            self._client = self._activate(com)
            self._initialise_client(com)
            capture = self._client.GetService(
                com.IAudioCaptureClient._iid_)
            # GetService returns IUnknown; query the capture interface.
            self._capture = capture.QueryInterface(
                com.IAudioCaptureClient)
            hr = self._client.Start()
            _check_hr(hr, "IAudioClient.Start")
            self._started = True
        except Exception:
            self.close()
            raise

    def _coinit(self) -> None:
        hr = coinit_mta()
        if hr == 0:
            self._com_initialised = True
        elif (hr & 0xFFFFFFFF) == RPC_E_CHANGED_MODE:
            # Thread already initialised in another apartment (e.g. an
            # STA UI thread). The handler is agile, so activation can
            # still proceed; do not uninitialise what we did not init.
            logger.warning(
                "COM already initialised on this thread (STA?) — "
                "continuing; per-app activation needs an agile handler")
            self._com_initialised = False
        else:
            _check_hr(hr, "CoInitializeEx")

    def _activate(self, com: SimpleNamespace):
        """Activate an IAudioClient for the target process tree."""
        params, pv = build_activation_params(self.pid)
        handler = com.ActivationHandler()
        self._handler = handler  # keep alive until activation completes
        handler_iface = handler.QueryInterface(
            com.IActivateAudioInterfaceCompletionHandler)
        op = ctypes.POINTER(
            com.IActivateAudioInterfaceAsyncOperation)()
        hr = com.mmdevapi.ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            ctypes.byref(com.IAudioClient._iid_),
            ctypes.byref(pv),
            handler_iface,
            ctypes.byref(op))
        _check_hr(hr, "ActivateAudioInterfaceAsync")
        # The async operation is ours to release (NAudio #878: leaking
        # it prevents re-activation). Dropping the reference lets
        # comtypes release it promptly after activation completes.
        self._async_op = op
        if not handler.event.wait(timeout=self.activation_timeout):
            raise PerAppCaptureError(
                "ActivateAudioInterfaceAsync timed out after "
                f"{self.activation_timeout}s waiting for the completion "
                f"handler (pid {self.pid}).")
        self._async_op = None
        if handler.activate_hr not in (0, None):
            _check_hr(handler.activate_hr, "Audio client activation")
        if handler.activated is None:
            raise PerAppCaptureError(
                "Audio client activation completed but returned no "
                f"interface (pid {self.pid}).")
        client = handler.activated.QueryInterface(com.IAudioClient)
        self._handler = None  # activation done; Windows is finished with it
        return client

    def _initialise_client(self, com: SimpleNamespace) -> None:
        """Initialize the client: requested format, else mix format."""
        wfx = com.WAVEFORMATEX()
        wfx.wFormatTag = WAVE_FORMAT_PCM
        wfx.nChannels = self.channels
        wfx.nSamplesPerSec = self.sample_rate
        wfx.wBitsPerSample = 16
        wfx.nBlockAlign = self.channels * 2
        wfx.nAvgBytesPerSec = self.sample_rate * self.channels * 2
        wfx.cbSize = 0
        try:
            hr = self._client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                self.buffer_duration_100ns,
                0,
                ctypes.byref(wfx),
                None)
            _check_hr(hr, "IAudioClient.Initialize")
            self._mix_format = None
            self._block_align = self.channels * 2
        except Exception as exc:
            # comtypes raises COMError on FAILED HRESULTs; only the
            # unsupported-format case deserves a retry with the mix
            # format — any other failure is real and must surface.
            hresult = getattr(exc, "hresult", None)
            if hresult is None or (
                    hresult & 0xFFFFFFFF) != AUDCLNT_E_UNSUPPORTED_FORMAT:
                raise
            logger.warning(
                "Requested format (48 kHz mono int16) rejected "
                "(AUDCLNT_E_UNSUPPORTED_FORMAT); falling back to the "
                "mix format with conversion.")
            # comtypes returns ["out"] params; they are not passed.
            mix_ptr = self._client.GetMixFormat()
            try:
                fmt = mix_ptr.contents
                self._mix_format = {
                    "rate": fmt.nSamplesPerSec,
                    "channels": fmt.nChannels,
                    "format_tag": fmt.wFormatTag,
                    "bits": fmt.wBitsPerSample,
                }
                self._block_align = fmt.nBlockAlign
                hr = self._client.Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_LOOPBACK,
                    self.buffer_duration_100ns,
                    0,
                    mix_ptr,
                    None)
                _check_hr(hr, "IAudioClient.Initialize (mix format)")
            finally:
                ctypes.windll.ole32.CoTaskMemFree(mix_ptr)

    # ------------------------------------------------------------ read ----
    def read_chunk(self, timeout: float = 0.1) -> bytes | None:
        """Return one packet as 48 kHz mono int16 bytes, or None on timeout."""
        if not self._started or self._capture is None:
            raise PerAppCaptureError(
                "read_chunk() called before start() completed")
        com = self._com
        deadline = time.monotonic() + timeout
        while True:
            # comtypes returns ["out"] params; they are not passed.
            n_frames = self._capture.GetNextPacketSize()
            if isinstance(n_frames, tuple):
                n_frames = n_frames[0]
            if n_frames:
                break
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)
        # GetBuffer returns its five ["out"] params as a tuple.
        data_ptr, frames, flags, dev_pos, qpc_pos = (
            self._capture.GetBuffer())
        try:
            n_bytes = frames * self._block_align
            if flags & AUDCLNT_BUFFERFLAGS_SILENT:
                raw = bytes(n_bytes)
            elif flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY:
                logger.warning("Per-app capture: data discontinuity "
                               f"({frames} frames)")
                raw = ctypes.string_at(data_ptr, n_bytes)
            else:
                raw = ctypes.string_at(data_ptr, n_bytes)
        finally:
            self._capture.ReleaseBuffer(frames)
        if self._mix_format is None:
            return raw
        return _convert_mix_chunk(
            raw, rate=self._mix_format["rate"],
            channels=self._mix_format["channels"],
            format_tag=self._mix_format["format_tag"],
            bits=self._mix_format["bits"],
            target_rate=self.sample_rate)

    # ------------------------------------------------------------ stop ----
    def stop(self) -> None:
        if self._client is not None and self._started:
            try:
                hr = self._client.Stop()
                _check_hr(hr, "IAudioClient.Stop")
            except Exception:
                logger.exception("IAudioClient.Stop failed")
            self._started = False

    def close(self) -> None:
        self._capture = None
        self._client = None
        self._handler = None
        self._async_op = None
        if self._com_initialised and self._com is not None:
            try:
                self._com.comtypes.CoUninitialize()
            except Exception:
                pass
            self._com_initialised = False
        self._com = None


# --------------------------------------------------------------------------
# Capture class — behind the existing WASAPICapture interface
# --------------------------------------------------------------------------
class ProcessLoopbackCapture(WASAPICapture):
    """System track from one process only (WASAPI process loopback).

    A :class:`WASAPICapture` whose hardware path is a
    :class:`_ComProcessLoopbackStream` for the target process tree
    (``mode == "per-app"``). Activation happens synchronously in
    :meth:`start` — a failure raises :class:`PerAppCaptureError`
    before any audio is written, so :class:`DualCapture` can fall back
    to system-wide loopback in ``"auto"`` mode instead of recording a
    silent track. When a ``chunk_source`` is supplied (tests), the
    hardware path is skipped exactly like the base class.
    """

    kind = "system"

    def __init__(self, output_path: str | Path, *,
                 process_name: str = "ms-teams.exe",
                 process_names: list[str] | None = None,
                 stream_factory=None,
                 **kwargs) -> None:
        super().__init__(output_path, **kwargs)
        self.process_name = process_name
        self.process_names = process_names or [process_name]
        #: Test seam: ``stream_factory(pid=..., sample_rate=...,
        #: channels=...)`` returning a stream with ``start`` /
        #: ``read_chunk`` / ``stop`` / ``close``. ``None`` = the real
        #: COM stream.
        self._stream_factory = stream_factory
        self._stream = None
        self._pid: int | None = None
        self.mode = MODE_PER_APP
        self.warning: str | None = None
        # Resolve the target now (same as PerAppCapture): the PID is
        # needed before activation, and the UI can report "not running"
        # before Record is pressed. The session's pid wins over the
        # tree root — it is the process actually rendering audio.
        info = resolve_capture_pid(self.process_names)
        self.process_info = info
        if info is not None:
            self._pid = info["pid"]

    # ------------------------------------------------------------ control -
    def _require_hardware(self) -> None:
        """Pre-flight checks (called synchronously from start()).

        Replaces the base class's pyaudiowpatch check: per-app capture
        needs Windows build 20348+ and pycaw, plus a running target
        process. Raises :class:`PerAppCaptureError` with the concrete
        reason — DualCapture catches it for the auto fallback. An
        injected ``stream_factory`` (tests) stands in for hardware,
        so the platform checks are skipped in that case.
        """
        if self._stream_factory is not None:
            return
        available, reason = per_app_capture_available()
        if not available:
            raise PerAppCaptureError(
                f"Per-app capture unavailable: {reason}")
        if self._pid is None:
            # Re-resolve: the process may have started since construction.
            info = resolve_capture_pid(self.process_names)
            self.process_info = info
            if info is not None:
                self._pid = info["pid"]
        if self._pid is None:
            raise PerAppCaptureError(
                f"Target process '{self.process_name}' is not running — "
                "no process tree to capture. Start Teams (or pick the "
                "running app) and try again.")

    def start(self) -> None:
        if self.chunk_source is None:
            self._require_hardware()
            factory = self._stream_factory or _ComProcessLoopbackStream
            self._stream = factory(
                pid=self._pid, sample_rate=self.sample_rate,
                channels=self.channels)
            try:
                self._stream.start()
            except Exception:
                self._stream = None
                raise
            # Warn (don't fail) when the target is running but not
            # rendering audio yet — activation succeeds and the stream
            # yields silence until it does. The level meter shows it,
            # and this warning says why.
            session = describe_target_session(self._pid)
            if session is not None and not session["active"]:
                self.warning = (
                    f"'{self.process_name}' (pid {self._pid}) is running "
                    "but has no active audio session yet — the system "
                    "track will be silent until it starts playing audio "
                    "(e.g. when the meeting starts).")
                logger.warning("ProcessLoopbackCapture: %s", self.warning)
            elif session is not None and session.get("peak") == 0.0:
                self.warning = (
                    f"'{self.process_name}' (pid {self._pid}) has an "
                    "active audio session but is currently silent — the "
                    "system track stays silent until it plays audio.")
                logger.warning("ProcessLoopbackCapture: %s", self.warning)
        try:
            super().start()
        except Exception:
            self._close_stream()
            raise

    def stop(self) -> Path:
        try:
            return super().stop()
        finally:
            self._close_stream()

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            logger.exception("Per-app stream stop failed")
        try:
            stream.close()
        except Exception:
            logger.exception("Per-app stream close failed")

    # ------------------------------------------------------------ internals -
    def _run_hardware(self) -> None:
        """Read loop for the already-started loopback stream."""
        if self._stream is None:
            raise PerAppCaptureError(
                "Per-app stream was not started (start() must run first).")
        while not self._stop_event.is_set():
            chunk = self._stream.read_chunk(timeout=0.1)
            if chunk:
                self._handle_chunk(chunk)
