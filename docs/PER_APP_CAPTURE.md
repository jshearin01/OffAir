# Per-app audio capture (Windows)

**Status (Nov 8, 2026): VERIFIED on real Windows (Windows 11 VM).**
First real run Nov 8: a 440 Hz tone player as the target process —
per-app capture (direct stream and high-level class) and the
system-wide loopback fallback each captured a clean 440.0 Hz tone
(rms ~11580, peak 16384, 99.5–99.8% nonzero), and
`per_app_capture_available()` returns `(True, None)`. Two real-Windows
bugs were found and fixed in the same run: the
`AUDCLNT_STREAMFLAGS_LOOPBACK` constant was `0x00000002`
(`AUDCLNT_BUFFERFLAGS_SILENT`, a GetBuffer flag), which made
`Initialize` fail with `0x88890021`, and the comtypes calls passed
`["out"]` params as byref arguments instead of reading their return
values. The loopback stereo fix (capture.py) also landed here: a
loopback device delivers its native channel count, so the system track
is now downmixed instead of read as mono. A real Teams meeting is
still the open gate for the whole pipeline — see
[How to test on real Windows](#how-to-test-on-real-windows).
Auto-detect's COM probe now also runs on real Windows without errors
(same out-param fixes); a real Teams meeting start/stop is still the
open gate for auto-detect end to end.

## What changed

Before this change, OffAir's "PerAppCapture" found the Teams process but
recorded **system-wide** WASAPI loopback (`mode:
"system-loopback-fallback"` in `metadata.json`) — every app playing sound
landed on the system track. pyaudiowpatch / PortAudio expose only
system-wide loopback.

Now the system track can be **true per-app capture**: only the target
process's audio (Teams by default), via the Windows Application Loopback
API. The old path is kept as an automatic fallback.

## Mechanism (verified from primary sources)

| Fact | Source |
|---|---|
| `ActivateAudioInterfaceAsync` on virtual device `VAD\Process_Loopback` with `AUDIOCLIENT_ACTIVATION_PARAMS` / `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS` (`TargetProcessId` + `PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`); requires **Windows 10 build 20348+**; target with no rendering streams yields silence; capture is not tied to a specific endpoint | Microsoft Learn, "Application loopback audio capture" sample |
| OBS 28+ uses exactly this for "Application Audio Capture" | `plugins/win-wasapi/win-wasapi.cpp` (OBS source) |
| GStreamer `wasapi2src` ships it; its `Wasapi2ActivationHandler` answers `QueryInterface` for `IUnknown` / `IAgileObject` / `IActivateAudioInterfaceCompletionHandler`, copies `audioclientactivationparams.h` (`VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`, modes 0/1), and notes `GetDevicePeriod` returns `E_NOTIMPL` on process loopback | GStreamer commits `7701b279` / `ba41200c` |
| Chromium's low-latency input path does the same activation for process loopback | `media/audio/win/audio_low_latency_input_win.cc` |
| Pure Python works (ctypes + comtypes): params in a `PROPVARIANT` `VT_BLOB`, async activation with a completion handler; the handler **must also implement `IAgileObject`** or activation returns `E_ILLEGAL_METHOD_CALL` with no explanation; `Initialize` accepts the requested format directly (16 kHz mono int16 in their case — we request 48 kHz mono int16) | Voxis, dev.to "Translating Windows system audio in real time — driverless, with no virtual cable" (2026-06-25) |
| Same `IAgileObject` trap, independently: a handler aggregating only `IMarshal` still failed; answering `QueryInterface(IAgileObject)` fixed it | Sunshine/Hydra issue #844 probe (2026-10-19 update) |
| Counter-report: one project found `comtypes.COMObject` did **not** answer `IAgileObject` for them and rewrote the handler in pure ctypes — see [Known risks](#known-risks) | AutoRecord-AVSI `WinPort/README_WINDOWS_EN.md` |
| `IAudioCaptureClient::GetBuffer` takes **5** parameters (`ppData, pNumFramesToRead, pdwFlags, pu64DevicePosition, pu64QPCPosition`) — a 4-parameter definition causes access violations | ReactOS `audioclient.idl`, Wine `audioclient.idl`, XDA anecdote |

### Verified constants (not guessed)

| Interface / value | IID / value |
|---|---|
| `IActivateAudioInterfaceCompletionHandler` | `41D949AB-9862-444A-80F6-C261334DA5EB` |
| `IActivateAudioInterfaceAsyncOperation` | `72A22D78-CDE4-431D-B8CC-843A71199B6D` |
| `IAgileObject` | `94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90` |
| `IAudioCaptureClient` | `C8ADBD64-E71E-48A0-A4DE-185C395CD317` |
| `IAudioClient` (from pycaw) | `1CB9AD4C-DBFA-4C32-B178-C2F568A703B2B` |
| `VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK` | `VAD\Process_Loopback` |
| `AUDCLNT_E_UNSUPPORTED_FORMAT` | `0x88890008` |

Each IID was cross-checked against **two independent sources**: the
`winapi` Rust crate (generated from the Windows SDK headers) and the
ReactOS/Wine IDL mirrors (plus go-wca and PortAudio for
`IAudioCaptureClient`). `IAgileObject` was confirmed in five places
(mingw-w64 `objidlbase.h`, ReactOS IDL, dotnet/wpf, kotlin-winrt,
Microsoft Learn).

### pycaw — what it provides and what it doesn't

pycaw **20260927** (verified by downloading the wheel and inspecting it)
defines the battle-tested `IAudioClient` vtbl (SDK order: `Initialize`,
`GetBufferSize`, …, `GetService`), `WAVEFORMATEX`,
`IAudioSessionManager2` / `IAudioSessionControl2` /
`IAudioMeterInformation` (`GetPeakValue`), and
`AudioUtilities.GetAllSessions()` / `GetProcessSession(pid)`. It does
**not** define `IAudioCaptureClient` or the activation interfaces — those
are hand-defined in `app/audio/perapp_capture.py` from the SDK headers.

## What was built

| File | Change |
|---|---|
| `app/audio/perapp_capture.py` *(new)* | `ProcessLoopbackCapture` (`WASAPICapture` subclass, `mode == "per-app"`), `_ComProcessLoopbackStream` (real COM path), `per_app_capture_available()`, `resolve_target_pid()` (prefers the process-tree root), `select_system_capture_class()`, `describe_target_session()` (best-effort "is Teams rendering yet?" probe), `build_activation_params()` + the ctypes structs, `_convert_mix_chunk()` / `_resample_linear()` (mix-format fallback) |
| `app/audio/capture.py` | `DualCapture` takes `capture_system_mode`; default system class comes from the selector; `start()` falls back to system-wide loopback in `"auto"` mode when per-app start fails (mic keeps running); `"per-app"` mode propagates the failure |
| `app/config.py` | `capture_system_mode` setting (`"auto"` default / `"per-app"` / `"loopback"`), validated, persisted via the settings allowlist |
| `app/platform_info.py` | `per_app_capture_status()` helper (never raises) |
| `app/ui/main_window.py` | Passes the configured mode to `DualCapture`; metadata is written **after** `start()` so a fallback is recorded honestly; mode label shows the actual mode; device-refresh tooltip notes per-app availability |
| `app/ui/settings_dialog.py` | "System audio capture (Windows)" picker + explanatory hint |
| `requirements.txt` | `pycaw==20260927; sys_platform == "win32"` (pure Python; the app runs without it and falls back) |
| `tests/test_perapp_capture.py` *(new)* | 25 tests (constants, struct layout, PID resolution, availability, fake-stream capture, format conversion, DualCapture fallback swap, config round-trip, Settings dialog picker) |
| `tests/test_capture.py` | Dual-capture test pins `capture_system_mode="loopback"` so it stays correct on Windows |

### Settings

- **Auto** (default) — try Teams-only per-app capture; on any failure
  (unavailable OS/build/package, Teams not running, activation error)
  fall back to system-wide loopback with a warning. The recording is
  never lost to an activation failure.
- **Teams only** — per-app capture or a loud failure. Use this when a
  silent fallback would be worse than no recording.
- **System-wide loopback** — always the old behaviour (pre-Nov 8).

`metadata.json` records the mode that actually ran: `"per-app"` vs
`"system-loopback-fallback"`.

### Design decisions worth knowing

- **Activation is synchronous in `start()`**, before the read thread
  starts — a failure raises `PerAppCaptureError` immediately instead
  of recording a silent track.
- **Format**: the client is initialised with 48 kHz mono int16
  directly (Voxis reports this works). On
  `AUDCLNT_E_UNSUPPORTED_FORMAT` it retries with `GetMixFormat` and
  converts (float32/float64/int32/int16, mono/stereo, linear resample
  when the rate differs) — so a target endpoint whose mix format is
  44.1 kHz (e.g. Remote Audio over RDP) is handled either way: the
  engine resamples the requested 48 kHz stream, or the mix-format
  fallback converts it. The system-wide loopback fallback
  (`pyaudiowpatch` path) negotiates the same way: it pre-flights the
  device's supported rates with `is_format_supported` and opens at
  the native rate (44.1 kHz when that is all the device offers),
  resampling to 48 kHz — it never hands PortAudio a refused format
  (Nov 10 2026: a fixed 48 kHz open on a 44.1 kHz-only loopback
  device crashed the frozen app; see `app/audio/capture.py`,
  `negotiate_stream_format`). Every `PyAudio()` construction /
  `terminate()` on that path is serialised through
  `app.audio.devices.PYAUDIO_INIT_LOCK`: `Pa_Initialize` is not
  thread-safe, and the mic thread constructing `PyAudio` while the
  calling thread negotiated the system format crashed the process
  with an access violation (0xc0000005, Nov 10 2026, reproduced
  twice) — the pre-flight alone did not remove the race.
- **Read loop polls `GetNextPacketSize`** (5 ms sleep) instead of an
  event handle — `GetDevicePeriod` returns `E_NOTIMPL` on process
  loopback, so there is no reliable period to wait on. A `SILENT`
  buffer flag yields digital zeros.
- **COM apartment**: the capture thread calls
  `CoInitializeEx(COINIT_MULTITHREADED)`; `RPC_E_CHANGED_MODE` is
  tolerated (the handler is agile). The async-operation object is
  released promptly after activation (NAudio #878: leaking it prevents
  re-activation).
- **Target PID**: when several processes share the exe name, the
  **tree root** (parent is not also a target process) is preferred,
  because `INCLUDE_TARGET_PROCESS_TREE` captures the root and its
  children.
- **Silent-target warning**: if Teams is running but has no active
  audio session yet (or an active but silent one), recording still
  starts — the track stays silent until Teams plays audio — but the
  session warning says why. The level meter shows the same thing.

## Failure modes

| Situation | Behaviour |
|---|---|
| Not Windows / build < 20348 / pycaw missing | Per-app unavailable; `"auto"` uses loopback, `"per-app"` fails with the concrete reason |
| Teams not running | No PID → `PerAppCaptureError`; `"auto"` falls back |
| Activation fails (`E_ILLEGAL_METHOD_CALL`, timeout, …) | Error names the stage + HRESULT; `"auto"` falls back |
| Teams running, no call yet | Capture runs; system track silent until Teams renders; warning recorded |
| Teams exits mid-recording | Stream yields silence / errors; the mic track is unaffected; stop the recording and re-start. With auto-detect on, the watcher ends the recording when the session disappears and starts a new one if a new meeting begins. |
| Target renders to a non-default output device | Capture still works (process loopback is not endpoint-tied); only the `describe_target_session` probe misses it |

## How to test on real Windows

Prerequisites: Windows 10 build 20348+ (check with `winver`) or
Windows 11; Python 3.10+; this repo; Teams desktop client.

1. `pip install -r requirements.txt` (gets `pycaw` + `pyaudiowpatch`).
2. Availability probe (no audio needed):
   `python -c "from app.audio.perapp_capture import per_app_capture_available; print(per_app_capture_available())"`
   → expect `(True, None)`.
3. Start Teams and join a test meeting (a second device, or a colleague).
   **At the same time, play a YouTube video in a browser** — this is the
   isolation test: the video must NOT appear on the system track.
4. In OffAir: Settings → System audio capture → Auto (default). Press
   Record, speak into the mic, let the other side speak for ~30 s.
5. Stop. Check `metadata.json`: `mode` must be `"per-app"`. Listen to
   `system.wav`: it must contain **only the remote side** (no YouTube
   audio). `mic.wav` is you; `combined.wav` is the mix.
6. Negative tests:
   - Teams **not running** → Record → warning + `mode`
     `"system-loopback-fallback"` (auto fallback works).
   - Settings → **Teams only**, Teams not running → Record fails with a
     clear error (no silent fallback).
   - Teams running but **idle** (no call) → Record → warning "no active
     audio session yet"; system track silent until the call starts.
   - **44.1 kHz loopback device** (e.g. Remote Audio over RDP, or any
     endpoint whose mix format is 44.1 kHz): Settings → System-wide
     loopback → Record → recording starts, `system.wav` is 48 kHz mono
     with the pitch intact, and `metadata.json` records
     `system_stream_sample_rate: 44100`. Before the Nov 10 2026 fix
     this crashed the app on Record.
   - Play audio in another app **during a call** → it must not leak into
     `system.wav`.
7. If activation fails with `0x8000000E` (`E_ILLEGAL_METHOD_CALL`),
   report it immediately — that implicates the comtypes-handler
   question (see Known risks); the pure-ctypes handler is plan B.

## What is still unverified

- The real WASAPI run (activation → `Initialize` → capture loop) has
  never executed — no Windows audio stack exists on the Linux VM.
  Everything COM-shaped is exercised only through injected fake streams.
- Whether `Initialize` accepts 48 kHz mono int16 directly on real
  hardware (Voxis says yes for 16 kHz; the mix-format fallback covers
  a rejection, but it is itself untested on hardware).
- Whether `comtypes.COMObject` answers `QueryInterface(IAgileObject)`
  on the target machine (see Known risks).
- Real-world behaviour when Teams switches output devices mid-call,
  and when the Teams process tree root is not the exe that renders
  audio (e.g. audio rendered by a child `ms-teams.exe` — covered by
  `INCLUDE_TARGET_PROCESS_TREE`, but unproven).

## Known risks

- **comtypes vs `IAgileObject`**: Voxis (shipping app) proves the
  `comtypes.COMObject` + `IAgileObject` in `_com_interfaces_` pattern
  works. AutoRecord-AVSI reports the opposite (their `comtypes`
  handler did not answer `IAgileObject`, so they rewrote it in pure
  ctypes with a hand-rolled vtable). The two reports conflict and the
  difference is unresolved without a Windows run. If the first real
  test fails with `E_ILLEGAL_METHOD_CALL`, the fix is a pure-ctypes
  handler (same interfaces, same GUIDs, no comtypes) — the rest of
  the pipeline is unchanged.
- **Voxis license note**: the Voxis repo is published under an
  all-rights-reserved notice (readable, not licensed for use). Nothing
  was copied from it — the implementation here was written from the
  Microsoft API documentation and the SDK-derived constants, with
  Voxis/GStreamer/OBS/Chromium used as corroborating references for
  the mechanism.

## Auto-detect + auto-record (implemented Nov 8 2026 — probe verified on real Windows)

**Status: built; the COM probe now runs on real Windows without errors**
(a 440 Hz tone player as the target: session found, live peak 0.49997
via `describe_target_session`, no `probe_error`). Before the Nov 8
out-param fixes the probe always failed on real Windows, so auto-record
could never have triggered. A real Teams meeting start/stop is still
the open gate for auto-detect end to end; see
[How to test auto-detect on real Windows](#how-to-test-auto-detect-on-real-windows).
The design below is implemented faithfully in
`app/audio/auto_detect.py` (`MeetingWatcher`), wired into the main
window and Settings.

### What was built

| File | Change |
|---|---|
| `app/audio/auto_detect.py` *(new)* | `MeetingWatcher` (1 Hz background thread, `poll_once()` state machine, fake probe + fake clock seams), `SessionSnapshot`, `auto_detect_available()`; real Windows probe enumerating **all** render endpoints via `IMMDeviceEnumerator` + `IAudioSessionManager2`/`IAudioSessionControl`/`IAudioMeterInformation` (pycaw/comtypes), mic-claim check on capture endpoints |
| `app/config.py` | `auto_detect_teams: bool = False` (**opt-in, default OFF**), gate constants (`AUTO_DETECT_POLL_INTERVAL_SECONDS` 1.0, `AUTO_DETECT_SUSTAIN_SECONDS` 2.5, `AUTO_DETECT_INACTIVE_GRACE_SECONDS` 60.0, `AUTO_DETECT_SILENCE_FUSE_SECONDS` 600.0, `AUTO_DETECT_ACTIVE_MIC_SECONDS` 30.0, `AUTO_DETECT_PEAK_THRESHOLD` 0.01, `AUTO_DETECT_SILENCE_PEAK` 0.001), settings round-trip |
| `app/platform_info.py` | `auto_detect_status()` helper (never raises) |
| `app/ui/settings_dialog.py` | "Start recording automatically when a Teams meeting starts" checkbox + consent note (OffAir does NOT trigger Teams' own recording banner); first-enable confirmation dialog; checkbox disabled with the reason on Linux |
| `app/ui/main_window.py` | Watcher starts after the window loads when enabled; start gate → `on_record(auto=True)` through the existing `DualCapture` path (respects `capture_system_mode`); stop gate → `on_stop(auto=True)` (flows into the auto-process pipeline); **AUTO** badge on the Record card + "Auto-recording — Teams meeting detected" status for the whole auto-recording; manual stop suppresses re-trigger until the session goes inactive; a manual recording is never stopped by the watcher; watcher stopped in `closeEvent` |
| `app/ui/theme.py` | `autoBadge` style (same red as the recording pill) |
| `tests/test_auto_detect.py` *(new)* | 36 tests: all gates with scripted snapshots + fake clock, secondary rule, manual suppression, probe-error skips, config round-trip, Settings consent paths, AUTO badge + watcher wiring (fake watcher + fake capture) |

### Gates (as built — stop gate revised Nov 8 2026)

- **Start — BOTH required.** Target session Active **and** peak ≥ 0.01
  sustained for 2.5 s (the design's 2–3 s window, midpoint). Process
  running alone never starts anything; a notification blip or a
  ringtone burst is shorter than the debounce.
- **Stop — ANY. Silence mid-meeting NEVER stops a recording.**
  Real meetings pause longer than 20 s (silent presenting,
  thinking, reading), so the design's 20 s silence rule was removed
  Nov 8 2026. Stop only on:
  - the session going Inactive/Expired (or disappearing) and
    **staying** that way for the 60 s grace (`AUTO_DETECT_INACTIVE_GRACE_SECONDS`) — a single inactive poll never
    stops anything, and the session coming back Active cancels the
    grace (transient stream renegotiation mid-call);
  - the target process exiting (immediate);
  - the long-silence fuse: session Active with peak ≤ 0.001 **and**
    the mic unclaimed, continuously for 10 min
    (`AUTO_DETECT_SILENCE_FUSE_SECONDS`) — a stuck/leftover
    session, not a real pause. Any audio, or the mic being claimed,
    resets the fuse.
  The gate rests on **session state** being the meeting-in-progress
  signal: Teams holds its audio session Active for the whole call
  (it keeps rendering, even silence, through the open stream), so
  peak level is only the start debounce.
- **Secondary start rule.** Session Active for > 30 s with the mic
  claimed starts a recording even with no peak (all-muted meeting).
  Mic-claim is a corroborating vote only, never the sole signal.
- **Manual-stop suppression.** After the user stops a recording by
  hand, the watcher does not re-trigger on the still-active session
  — it waits until the session goes inactive/absent first. An
  auto-stop re-arms immediately, so a new meeting starts a new
  recording.
- **Probe failures are skipped, not misread.** A snapshot with
  `probe_error` set (COM enumeration failed) or a probe that raises
  leaves state and counters untouched — a COM hiccup mid-meeting
  never ends the recording.

### Deviations from the design (deliberate)

- **Sustain is 2.5 s**, the midpoint of the design's "~2–3 s" —
  tunable in `app/config.py`.
- **The 20 s silence stop was removed Nov 8 2026:** real meetings
  pause longer than 20 s, so silence mid-meeting never stops a
  recording. The stop gate is now session-state driven (60 s
  inactive grace / process exit / 10-min silence fuse) — see
  Gates (as built) above.
- **Silence peak is 0.001**, not literally 0.0 — a session meter
  reports exactly 0.0 when silent, and 0.001 leaves headroom for
  meter noise without admitting quiet speech (start needs 0.01).
- **Pre-roll core + feeder built Nov 21 2026; UI wiring shipped
  in 1.0.5.** The ring buffer, the drain-on-start hook, the
  capture prepend, the always-on feeder, and the opt-in config
  are built (see "Pre-roll core + feeder" below), and the
  Settings toggle + consent note wire them to the UI. Pre-roll
  stays opt-in and default OFF, a privacy decision of its own.
- **No network / window-title / log fallbacks were built.** The
  session signal is the only signal; the alternatives stay
  documented below as evaluated-and-rejected.

### Residual gaps (unchanged from the design)

- **The stop gate assumes Teams keeps its audio session Active
  across quiet stretches — this MUST be verified on a real
  Windows Teams call.** Watch the session state across a 60+ s
  silent stretch (see the test procedure below): if Teams holds
  the session Active through silence, the gate is right; if Teams
  actually closes the session between utterances, the session will
  flap Inactive mid-meeting, the 60 s grace will absorb short
  gaps, but a quiet stretch longer than 60 s would end the
  recording — and the design would need revisiting (e.g. a much
  longer grace, or a different meeting signal). The real-Windows
  test must say plainly which behaviour Teams shows.
- **All-muted, mic-unclaimed silent meeting will not auto-start**
  (the secondary rule needs the mic claimed). Accepting that gap
  for now; the alternative — starting on session-Active alone —
  would record every Teams session that sits Active while silent.
- **Linux is unsupported, explicitly.** The watcher is Windows-only
  (WASAPI sessions); the Settings checkbox is disabled with the
  reason on Linux instead of pretending it works. No fake Linux
  detection was attempted.
- **The real probe is untested on Windows.** The pycaw interface
  names used (`IMMDeviceEnumerator`, `IAudioSessionManager2`,
  `IAudioSessionControl.GetState/GetProcessId`,
  `IAudioMeterInformation.GetPeakValue`) are the documented pycaw
  API, but pycaw is not installed on this Linux VM so they could
  not be verified against the installed library — if a name is
  wrong, the probe reports `probe_error` and the watcher logs it
  instead of crashing. The first real run will confirm or correct
  them.

### How to test auto-detect on real Windows

Prerequisites: same as per-app capture (Windows 10 build 20348+
is not required for detection itself — session enumeration predates
process loopback — but the test below uses per-app capture too, so
keep the same bar), Teams desktop client, a second device or a
colleague for the meeting.

1. `pip install -r requirements.txt` (gets `pycaw`).
2. Availability probe: `python -c "from app.audio.auto_detect import auto_detect_available; print(auto_detect_available())"` → expect `(True, None)`.
3. In OffAir: Settings → tick "Start recording automatically when
   a Teams meeting starts", confirm the consent dialog, OK.
   The status bar should read "Auto-detect on — watching for a
   Teams meeting to start."
4. Join a Teams meeting from the second device. Within ~3 s of
   someone speaking, OffAir should start recording on its own:
   the **AUTO** badge appears on the Record card and the status
   reads "Auto-recording — Teams meeting detected".
5. Talk, then all go silent for 60+ s: the recording must KEEP
   going (silence mid-meeting never stops it). This step is also
   the check on the stop gate's key assumption — that Teams keeps
   its audio session Active through silence. If the recording
   instead stops ~60 s into the silence, Teams is closing the
   session between utterances and the design needs revisiting
   (report it plainly; do not just lengthen the grace). To check
   the stop gate, end the meeting: the recording should stop by
   itself once the Teams session has been inactive for the 60 s
   grace, and auto-processing should run. (The 10-min silence
   fuse is covered by unit tests, not by sitting silently for
   10 min.)
6. Start a second meeting: it should auto-record again.
7. Negative tests:
   - Teams running all day, no meeting → **no recording starts**
     (leave it watching for an hour; check no session folders appear).
   - Play a notification sound / receive a chat message while
     Teams is idle → no recording.
   - Have someone ring you and decline → no recording, or at most
     a recording that auto-stops once the session has been
     inactive for the 60 s grace.
   - During an auto-recording, press **Stop** by hand → the
     recording ends and does **not** restart while the meeting
     continues; only a new meeting starts a new recording.
   - Start a recording by hand during a meeting, then end the
     meeting → the manual recording keeps going (the watcher never
     stops it).
   - All-muted meeting, nobody speaks, mic unclaimed → no
     auto-start (known gap); unmute and speak → starts within ~3 s.
   - Teams on a Bluetooth headset (non-default output device) →
     still detected (all-endpoint enumeration); report if not.

Task: start recording automatically when a Teams meeting starts, stop
when it ends. The Teams process runs all day, so "process is running"
is **not** a meeting signal. Evaluated signals, best first:

### 1. Teams audio session becoming active with flowing audio (recommended)

Same COM layer as the capture backend. Enumerate render sessions
(`IAudioSessionManager2` → `IAudioSessionEnumerator` →
`IAudioSessionControl2`; pycaw already exposes all of this plus
`AudioUtilities.GetAllSessions()` and `IAudioMeterInformation.GetPeakValue`),
find the session whose process is the Teams tree root, and watch:

- **Start gate**: session state becomes **Active** *and* peak level
  stays above a small threshold (e.g. > 0.01) for a sustained window
  (~2–3 s). State alone is not enough — a session can sit Active
  while silent; peak alone is not enough — a notification blip has
  peak too. Both together, debounced, is the meeting signal.
- **Stop gate**: session becomes Inactive/Expired and stays that
  way for a 60 s grace (covers transient renegotiation; a single
  inactive poll never stops anything), **or** the Teams process
  exits, **or** the long-silence fuse fires (session Active with
  peak ≈ 0 and the mic unclaimed for 10 min — a stuck/leftover
  session, not a real pause). Silence mid-meeting NEVER stops a
  recording (revised Nov 8 2026 — the original design's 20 s
  silence rule was wrong: real meetings pause longer than that).
  Session state is the meeting-in-progress signal; peak level is
  only the start debounce.
- **Poll**: 1 Hz polling on a background thread is plenty; session
  enumeration is cheap. Note pycaw's `GetAllSessions()` covers the
  **default render endpoint only** — the watcher must enumerate **all**
  render endpoints via `IMMDeviceEnumerator` (Teams on a Bluetooth
  headset is not on the default device), so it needs its own
  enumeration rather than reusing `GetAllSessions()` as-is.
- **Corroborating signal**: the mic capture endpoint's sessions
  (`IAudioSessionManager2` on the capture device) show whether Teams
  has claimed the mic. Useful as a second vote (you speaking), not a
  primary signal — you can be in a meeting muted and listening.

### 2. Teams network activity (second choice)

Meeting traffic is STUN/TURN UDP (ports 3478–3481). Detectable with
packet capture (npcap / WinDivert) or ETW. Heavier (needs a driver or
admin rights), more moving parts, and it answers "in a call" without
saying anything about audio flow. Keep as a fallback if the session
signal proves unreliable.

### 3. Meeting window title / class (fragile)

Win32 `EnumWindows` looking for a Teams meeting window. Titles and
window classes vary across the classic client, the new client, and
the PWA; a window can exist without audio, and a call can exist
minimised. Not recommended as the primary signal.

### 4. Teams logs (not recommended)

The client writes logs, but the format is version-fragile and
log-watching is not a real-time signal.

### False starts and how the gates handle them

| Event | What happens | Gate response |
|---|---|---|
| Incoming call ringing | Teams session Active + ringtone audio flows | Debounce (2–3 s sustained peak) filters short rings; a declined call ends the session → the recording (if one started) stops after the 60 s inactive grace |
| Notification sounds / toasts | Short audio bursts through the Teams session | Below the sustained-activity window — no start |
| Someone unmutes and says one word, then silence | Brief peak, then nothing | Start fires (correct — a meeting is happening); the recording keeps going through the silence until the session ends (or the 10-min fuse, if the session is left Active with no audio and the mic unclaimed) |
| Meeting where everyone is muted, nobody speaks | Session Active, peak 0 | Start gate (needs peak) never fires — **false negative**; mitigate by also starting on "session Active for > 30 s with the mic claimed" as a secondary rule, or accept missing silent meetings |
| You watch a video *inside* Teams (e.g. a shared clip) | Real Teams audio flows | Correct to record — it is a meeting |

### Pre-roll problem

With a 2–3 s debounce, the first syllables ("hello") are missed. Options:
(a) accept it; (b) keep the per-app stream running into a rolling
~30 s ring buffer and commit the buffer when the start gate fires
(more code, always-on capture — a privacy consideration in itself);
(c) start on session-Active with a 0 s gate and rely on the stop rule
to kill false starts (rings would start recordings — bad). Recommend
(a) first, (b) as a later phase if the miss matters.

### Pre-roll core + feeder (built Nov 21 2026)

Option (b)'s machinery exists, including the always-on feeder
that fills the buffer before Record is pressed:

- `app/audio/preroll.py` — `PreRollBuffer`: a per-track
  (`mic` / `system`) ring buffer of chunks, memory-only
  (no file I/O anywhere in the module; chunks carry no timestamps
  — prepend needs order and sample counts only, so none are
  stored). Feeding is the `feed`
  seam; draining is `drain_all()`, which returns
  `{"mic": samples, "system": samples}` and clears the buffer.
- `app/audio/auto_detect.py` — `MeetingWatcher` accepts an
  optional `preroll_buffer=` and drains it when the start gate
  fires; the UI layer takes the drained audio with
  `MeetingWatcher.take_preroll()` (one-shot, `{}` when empty).
- `app/audio/capture.py` — `DualCapture.start(preroll_samples=...)`
  writes the drained audio ahead of the first live chunk on each
  track (`WASAPICapture.set_preroll`); with no pre-roll, capture
  is byte-identical to before.
- `app/config.py` — `preroll_enabled` (default `False`) +
  `preroll_seconds` (default 30, range 10–120), persisted in
  settings. The exact consent text the Settings UI must show is
  `PREROLL_CONSENT_NOTE` in `app/config.py`: it says plainly the
  app captures before Record is pressed, the buffer is
  memory-only, and buffered audio is discarded if no recording
  starts.

- `app/audio/preroll_feeder.py` — `PreRollFeeder`: the feeder.
  While pre-roll is enabled and no recording is in progress, one
  background thread per track (mic / system) opens that track's
  stream in shared mode — WASAPI via pyaudiowpatch on Windows
  (format negotiation + resampling reused from `capture.py`),
  sounddevice / PortAudio on Linux (the same `InputStream` shape
  as `linux_capture.py`, devices from the shared enumeration in
  `app/audio/devices.py`) — and feeds the buffer continuously.
  Memory-only, never disk; no file I/O anywhere in the module.
  Device-open / stream failures degrade silently (logged only):
  that track's pre-roll is empty and recording still works.
  `start()` / `stop()` are idempotent and race-free; `stop()` is
  synchronous and returns only after the streams are closed.

Feeder lifecycle contract for the UI layer (U4) — call order
matters, and the feeder must never run while a recording is in
progress:

```
buffer = PreRollBuffer(seconds=config.preroll_seconds,
                       sample_rate=SAMPLE_RATE)
feeder = PreRollFeeder(buffer, enabled=config.preroll_enabled,
                       sample_rate=SAMPLE_RATE,
                       mic_device_index=<mic combo currentData or None>,
                       loopback_device_index=<loopback combo currentData or None>)
feeder.start()   # when pre-roll is on; no-op returning False when
                 # disabled or no usable device; never raises
# Manual Record handler:
feeder.stop()                    # synchronous: streams closed
preroll = buffer.drain_all()     # {} when nothing buffered
capture.start(preroll_samples=preroll)
# Auto-detect meeting-started handler:
feeder.stop()
preroll = watcher.take_preroll()
buffer.clear()  # discard anything fed between the gate's drain and
                # this stop(); it is meeting audio, not pre-roll
capture.start(preroll_samples=preroll)
# After DualCapture.stop() returns (and if a start failed):
feeder.start()  # resume only if still enabled
# Pre-roll turned off / app shutdown:
feeder.stop()
buffer.clear()
```

Device-conflict finding (honest): whether WASAPI shared mode
would allow the feeder's loopback stream and a recording's
loopback stream to be open at the same time was **not** verified
on real Windows hardware — no concurrent open is relied on, and
the stop-before-start contract above is mandatory regardless.
The system track is system-wide loopback / a monitor source, not
per-app isolated audio (the same limitation as the loopback
fallback track). Manual Record prepends only when the same opt-in
is on (the UI layer drains the same buffer and passes it to
`DualCapture.start`). The UI wiring shipped with the feature: the
Settings toggle + consent note and the warning dialogs are in the
app, and the UI layer drains the same buffer and passes it to
`DualCapture.start`.

Still open: a real Windows run of the feeder — open the mic +
loopback streams, feed for 30 s, stop, start a recording with the
drained pre-roll, and confirm the WAV leads with the buffered
seconds and the recording is otherwise unaffected. The feeder is
proven only on fake streams + synthetic audio and on the Linux
sounddevice path with a fake sounddevice (see
`tests/test_preroll.py`); no real Teams audio, real Windows
capture, or RDP-only check is claimed.

### Where the watcher belongs

On the **same session-enumeration layer** as the capture backend
(pycaw/COM), as a small polling component emitting `meeting_started` /
`meeting_ended` events that the recorder subscribes to. It shares the
COM initialisation and the Teams-PID resolution with the capture code,
and it is the only layer that can see both session state and peak level.

### Must be opt-in

Auto-recording is a privacy decision, not a default: it must be an
explicit opt-in setting (default OFF), with the recording indicator
unmissable while it runs. Recording laws also vary by jurisdiction —
the app already only records when the user presses Record; auto-record
extends that, so the opt-in and a first-run consent note are part of
the build, not an afterthought.
