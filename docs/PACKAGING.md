# Packaging OffAir as a standalone Windows executable

## Why

`pip install -r requirements.txt` can fail on a user's machine when a
package has no prebuilt wheel for their Python version — most
recently PyAV (`failed building wheel for AV`, asking for
"Microsoft Visual C++ 14.0" to build it from source). The frozen
executable sidesteps pip entirely: no Python, no pip, no MSVC build
tools on the target machine. The user runs `OffAir.exe` (or a
setup.exe that installs it) and the app just starts.

## What is bundled, and what is not

Built with PyInstaller 6.x from `OffAir.spec` (repo root).

| In the bundle | Notes |
|---|---|
| Python 3.12 + the `app/` package | Frozen from `main.py` |
| PyQt6 + Qt plugins (incl. QtMultimedia for sample playback) | Standard PyInstaller Qt hooks |
| faster-whisper + ctranslate2 + av | Transcription; native DLLs via `collect_all` |
| onnxruntime + speakeronnx | Semantic search + ONNX ECAPA embeddings |
| sherpa-onnx | Default diarization backend |
| soundfile, numpy, psutil, comtypes, pycaw, pyaudiowpatch | Capture / COM / utilities |
| Visual C++ runtime DLLs (vcruntime140.dll, msvcp140.dll, …) | PyInstaller bundles the ones found on the **build** machine |

| NOT bundled — downloads on first use, same as the pip install | Notes |
|---|---|
| Whisper `base` model (~142 MB) | Into `%USERPROFILE%\.offair\models` on first transcribe |
| sherpa diarization models (~33 MB) | Same folder, from public GitHub releases |
| speakeronnx ECAPA weights (~20 MB) | Hugging Face, no token |
| pyannote models | Only for the opt-in "Full" diarization mode (needs a HF token) |

| Deliberately EXCLUDED | Why |
|---|---|
| torch / torchaudio / lightning / pyannote.audio / speechbrain / scipy | Only needed for the opt-in "Full" diarization mode and the SpeechBrain embedding backend. The default path (sherpa + ONNX ECAPA + simple mode) never imports them, and every import site is inside a `try/except ImportError` with a guided error. Excluding them saves ~1.5 GB. Picking "Full" mode or the SpeechBrain backend in the frozen app shows that guided error instead of working — same as a pip install without those packages. |

Two data entries are bundled: the application icon
(`assets/offair.ico`, see "Application icon (1.0.1)" below) and the
`docs/` Markdown folder, rendered by the in-app Help view
(`app/ui/help_view.py`, resolved via `app/resources.docs_dir()`).
A frozen install without `docs/` still starts and the Help view
shows a plain "not included in this install" message per missing
file — but the spec ships it, so the Help view is the repo docs,
not a stale copy. Settings live in
`%USERPROFILE%\.offair\settings.json`, recordings in
`%USERPROFILE%\OffAir\recordings` — the exe reads/writes the same
places the pip install does.

## Build machine prerequisites

1. Windows 10/11 (the build is done on a Windows 11 VM;
   the exe targets the same Windows version family).
2. Python 3.12 venv with `pip install -r requirements.txt` done.
3. **Microsoft Visual C++ Redistributable (x64)** installed on the
   build machine — `https://aka.ms/vs/17/release/vc_redist.x64.exe`,
   `/install /quiet /norestart`. Without it, the native DLLs
   (onnxruntime, ctranslate2, torch) fail to load even in the venv
   with `WinError 126` / "DLL load failed", and PyInstaller has no
   runtime DLLs to bundle. (This is a *runtime*, not the build
   tools — no MSVC compiler is needed anywhere in this flow.)
4. `pip install pyinstaller` in the venv (6.22.3 used for the first
   build; pyinstaller-hooks-contrib comes with it).

## Rebuilding

```bat
cd C:\OffAir
.\.venv\Scripts\pyinstaller.exe --noconfirm OffAir.spec
```

`--noconfirm` is required on rebuilds: without it PyInstaller fails
(exit 1) when `dist\OffAir` already exists, and Inno Setup then
silently packages the stale dist — that exact mistake happened on the
1.0.1 rebuild (Nov 10) before the flag was added here.

Output is `dist\OffAir\` (onedir): `OffAir.exe` plus support files.
Copy the whole folder to the target machine and run `OffAir.exe`.

**`mcp[cli]` is required (1.0.2, Nov 11 2026):** the spec's
`collect_all("mcp")` imports every `mcp` submodule, including
`mcp.cli`, which calls `sys.exit(1)` at import time when `typer`
is missing. Plain `mcp>=2.0` therefore fails the PyInstaller
build after ~40 s. `requirements.txt` now says `mcp[cli]>=2.0`;
on a fresh build machine install that before building.

Smoke tests for a fresh build (from a plain cmd, no venv):

```bat
cd dist\OffAir
OffAir.exe --no-gui --list-devices
```

That must exit 0. It prints nothing to the console — since 1.0.1
the exe is a windowed build (`console=False`, see below), so there
is no console for it to print to. Instead the config + device list
is written to the log file: check
`%USERPROFILE%\OffAir\recordings\offair.log` for the `Output root`,
`Microphones`, `Loopback` and `Capture` lines. Then launch the GUI
once (`OffAir.exe`, no args) and confirm the main window opens —
and that no console window appears alongside it. For the heavy
paths, run a transcribe + a diarize from the UI once, or
import-check inside the frozen app — the first build's verification
is recorded below.

Frozen self-test (heavy imports + model downloads in the frozen
environment):

```bat
.\.venv\Scripts\pyinstaller.exe --noconfirm --onedir --name OffAirSelfTest --console --collect-all faster_whisper --collect-all ctranslate2 --collect-all av --collect-all onnxruntime --collect-all sherpa_onnx --collect-all speakeronnx --collect-all soundfile --collect-all certifi --exclude-module torch --exclude-module torchaudio --exclude-module pyannote --exclude-module speechbrain --exclude-module scipy --exclude-module lightning --exclude-module matplotlib --exclude-module tkinter selftest_frozen.py
cd dist\OffAirSelfTest
OffAirSelfTest.exe C:\OffAir\tests\fixtures\speaker-john-16k.wav
```

Exit 0 only if every step passes. `selftest_frozen.py` (repo root)
is build-verification tooling, not part of the shipped app.

Installer (after the onedir build):

```bat
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" OffAir.iss
```

Output: `installer\OffAir-Setup-1.0.6.exe`.

## Gotchas found during the first build (Nov 9 2026)

1. **The build machine needs the VC++ Redistributable, not just
   the app.** A bare Windows eval install has no `vcruntime140.dll`,
   so `onnxruntime`, `ctranslate2` and `torch` all failed to import
   in the venv itself (`WinError 126` / "DLL load failed") before
   any packaging. Installing `vc_redist.x64.exe` fixed every one.
   PyInstaller then bundles those runtime DLLs into the exe, so
   the *target* machine needs nothing — but the build machine must
   have the redistributable (prerequisite 3 above).
2. **The sherpa model downloader needed certifi.** On the same bare
   eval install, urllib's default verification failed for
   github.com's chain (`CERTIFICATE_VERIFY_FAILED`) while
   pypi.org/huggingface.co verified fine — a stale/incomplete
   system cert store. `app/diarization/sherpa.py` now verifies
   downloads against certifi's CA bundle when it is installed
   (explicitly declared in `requirements.txt`), falling back to
   the default context otherwise. The Whisper download never had
   this problem (huggingface_hub handles certificates itself).
   The frozen exe downloads both sherpa model files from scratch
   in ~4 s (verified in the self-test).

3. **PowerShell remoting mangles inline `python -c` quoting.** When
   driving the guest over SSH, a `python -c "..."` one-liner with
   escaped quotes gets its backslashes eaten and fails with
   `SyntaxError: unterminated string literal`. Write the snippet to
   a `.py` file, `scp` it to the guest, and run the file instead.
   (Same for any command with nested quoting — prefer a script file.)
4. **Frozen-content check (1.0.3, Nov 19 2026):** in this
   PyInstaller 6.x onedir layout the app modules' *names* are
   visible as plain strings inside `OffAir.exe` itself (the PYZ
   table of contents is appended to the exe), so `record_view`,
   `nearlive`, `chunker`, etc. can be confirmed with a raw byte
   search of the exe. String *constants* inside the `.pyc` files
   are compressed and will NOT match — a missing constant (e.g.
   `transcript-live`) proves nothing; check module names instead.
5. **PowerShell `$ErrorActionPreference = 'Stop'` aborts build
   scripts on native stderr.** PyInstaller logs INFO to stderr; under
   `Stop`, PowerShell treats native-command stderr as terminating
   and aborts the script on the first log line. Check `$LASTEXITCODE`
   explicitly instead (1.0.5 build, Nov 22 2026). scp quirk from the
   same round: guest downloads need forward-slash remote paths
   (`C:/OffAir/...`); backslashes fail, while uploads accept them.
6. **Detached builds die silently over SSH.** A build launched
   detached from the SSH session (e.g. via `Start-Process`
   without waiting) is killed with the session and leaves empty
   logs and no artifacts. Run builds in the foreground over SSH
   with `Start-Transcript` writing to a log file in the guest,
   so progress survives a dropped session (1.0.6 build,
   Nov 22 2026).

## No console window (1.0.1)

The 1.0.0 build was `console=True` so the frozen app could be
debugged from its console. 1.0.1 flips `OffAir.spec` to
`console=False`: double-clicking `OffAir.exe` (or the installed
app) shows only the app window and its icon, no console window —
a normal desktop app.

A windowed PyInstaller build has no console: `sys.stdout` and
`sys.stderr` are None, and any `print()` at startup raises
`AttributeError` and crashes the app before the window opens. The
startup-path audit for the flip (Nov 10 2026) found exactly four
stdout/stderr dependencies, all in `main.py` / logging — there is
no `print()` anywhere in `app/`:

* `main.py` `smoke_check()` — all of its lines now go through
  `_emit()`: print when a console stream exists, `logger.info`
  to `offair.log` when it does not. The `--no-gui` /
  `--list-devices` smoke check is a developer tool that is only
  meaningful with a console; in the windowed build it exits 0
  and the same lines land in the log file.
* `main.py` PyQt6-missing branch — the old `print(..., file=
  sys.stderr)` is now `_emit(..., error=True)`: in a windowed
  build the message goes to the log file (the crash dialog needs
  Qt, so the log is the only channel there). Returns exit code 2
  as before.
* `main.py` argument parsing — `argparse` prints usage to
  `sys.stderr` on a bad command line, which would raise in a
  windowed build. `main()` catches that failure, records it in
  the default log file, and exits 2 instead of crashing.
* `app/logging_setup.py` `setup_logging()` — the console
  `StreamHandler` is now attached ONLY when `sys.stderr` is not
  None. In a windowed build the log file is the only sink; with a
  console (dev runs, `--no-gui`) the console handler is attached
  as before. Uncaught exceptions still go through
  `install_exception_hook` to the log file + the crash dialog,
  never a bare print.

`selftest_frozen.py` needed no change: it is built as its own
`--console` exe, so its PASS/FAIL/RESULT prints have a console,
and it never asserted on `OffAir.exe --no-gui` stdout. The
frozen-app smoke test now asserts on exit code + log-file
contents instead of stdout text (see "Rebuilding" above).

Regression guards: `tests/test_packaging.py` — a spec-text test
for `console=False`, a `setup_logging` file-only test with
`sys.stderr` patched to None, a `main(["--no-gui"])` test with
both streams None (exit 0 + log lines), and source guards that
`main.py` prints only through `_emit` and the startup modules
have no `print()` at all.

## Application icon (1.0.1)

The 1.0.0 build used the PyInstaller default icon. 1.0.1 ships a
real one: a bold mark that reads at 16 px — a white microphone
(capsule + grille slits, cradle arc, stem and base) with a short
soundwave bar on each side, on an indigo rounded-square tile
(gradient #4F46E5 → #4338CA, the app accent from
`app/ui/theme.py`). No text, no fine detail.

* Master source: `assets/offair-icon.png` (1024×1024).
  Regenerate it (and the .ico) with `python3 assets/make_icon.py`
  — the drawing is Pillow code, so the icon is reproducible from
  the repo, not a one-off binary.
* Windows icon: `assets/offair.ico`, multi-size 16/32/48/256,
  built from the PNG by the same script. All four sizes were
  extracted back out and viewed during the 1.0.1 work: 256 px is
  the full mark, 48/32 px read as a white mic on the indigo tile,
  16 px reads as a white mic blob on indigo.
* Wired in three places:
  * `OffAir.spec` — `icon="assets/offair.ico"` in `EXE()` (the
    exe's icon in Explorer / the taskbar) and the .ico in the
    `datas` (so the running app can load it).
  * `OffAir.iss` — `SetupIconFile=assets\offair.ico` (the
    installer's icon).
  * In-app — `app/resources.py` resolves `assets/offair.ico`
    next to the repo root in source runs and under
    `sys._MEIPASS` when frozen; `main.py` sets it on the
    `QApplication` and `MainWindow.__init__` sets it on the
    window, so the title bar, taskbar and dialogs all show it.
* A missing icon file is never fatal: `set_app_icon()` logs a
  warning and returns False, and the app starts without its
  mark. Regression guards in `tests/test_packaging.py` assert the
  window + application icons are not null (Qt half) and that a
  missing file returns False without crashing.

## Measured results (first build, Nov 9 2026, Windows 11 VM)

Build machine: 2-vCPU VM, PyInstaller 6.22.3, Python 3.12.10.
Build time ~5 min (onedir), ~6 min (onefile), ~3 min (installer
compile, lzma2/ultra). The onedir/onefile rows below are the
1.0.0 measurements; the installer row is the current 1.0.6
installer — later bundles are bigger (2,026 files in 1.0.6, vs
1,161 in 1.0.0) because the MCP stack is now bundled.

| Artifact | Size | Cold start | Warm start |
|---|---|---|---|
| onedir `dist\OffAir\` (1,161 files) | 398 MB | ~13 s first launch, ~6 s after (Defender scan dominates) | 0.9 s (`--no-gui`) |
| onefile `dist\OffAirOnefile.exe` | 170 MB | ~46 s first launch | ~18 s (unpacks to %TEMP% on **every** launch) |
| installer `installer\OffAir-Setup-1.0.6.exe` | ~119 MB | — (installs the onedir bundle) | same as onedir once installed |

Verification (all on the Windows 11 VM, from a plain cmd with no
venv / no Python on PATH):

- `OffAir.exe --no-gui --list-devices` → exit 0, pyaudiowpatch
  available, 6 microphones, 1 loopback device, Teams not running.
  Same output from the onefile exe and from the installed exe.
- Frozen self-test (`selftest_frozen.py`, built as its own exe):
  **all steps PASS, exit 0** — heavy imports (faster-whisper, av,
  ctranslate2, onnxruntime, sherpa_onnx, speakeronnx, soundfile,
  QtMultimedia), Whisper tiny download + transcribe of a real clip
  (2 segments, RTF 0.10), sherpa model download from scratch
  (both files, ~33 MB, 4.0 s).
- GUI launch: the frozen exe starts and stays alive and
  responding (checked at 12 s and 20 s, no crash, no error dialog).
  The main window painting itself was **not visually confirmed**:
  the VM's console display is off/locked, so a screen capture came
  back black and a window-handle probe from the interactive session
  returned no result. A 30-second visual check over RDP is still
  owed (the Qt UI code itself is proven on this machine — the full
  test suite incl. Qt tests is green and the screenshots render).
- VC++ runtime DLLs (`vcruntime140.dll`, `msvcp140.dll`, …) are
  present in the bundle — the exe needs nothing installed on the
  target machine.

## onedir vs onefile

**Ship onedir (via the installer) as the default.** Onefile looks
tidier (one 170 MB file vs a 398 MB folder) but it unpacks the
whole bundle to a fresh `%TEMP%` folder on *every* launch, so
even a warm start costs ~18 s on this VM and it churns disk.
Onedir starts in ~1 s warm and the installer hides the folder
anyway. Onefile is only worth it for hand-a-file-to-someone
distribution, where a single file matters more than startup time.

## Installer (Inno Setup)

Feasible and verified. Inno Setup 6.7.1 (free, `OffAir.iss` in the
repo root) compiles the onedir bundle into `OffAir-Setup-1.0.6.exe`
(~119 MB, lzma2/ultra). The 1.0.2 bundle is bigger than 1.0.1's
(2,014 files vs 1,161) because the MCP stack (typer/rich/click)
is now bundled. Per-user install, no admin rights needed:
files go to `%LOCALAPPDATA%\Programs\OffAir`, with a Start-menu
entry, optional desktop shortcut, and a registered uninstaller.
Silent install tested in the VM (`/VERYSILENT /NORESTART`) and
the installed exe passes the `--no-gui` smoke test (exit 0).

Note: `jrsoftware.org/download.php/is.exe` no longer serves the
binary directly (it returns an HTML page); the direct link is the
GitHub release,
`https://github.com/jrsoftware/issrc/releases/download/is-6_7_1/innosetup-6.7.1.exe`.
Inno Setup itself is installed with
`innosetup-6.7.1.exe /VERYSILENT /NORESTART /SUPPRESSMSGBOXES`.

## Code signing

There is no signing certificate. An unsigned `OffAir.exe` (or
setup.exe) will trigger Windows SmartScreen ("Windows protected your
PC") on first run on a fresh machine; the user clicks "More info" →
"Run anyway". A code-signing certificate (OV, ~$200–400/year) would
remove that warning after the binary builds reputation. Not set up
for the first build.

## Release checklist (the routine flow, established across the 1.0.0–1.0.6 releases)

- [ ] Build onedir from a clean checkout, run the smoke tests above
- [x] Ship onedir via the installer (decision settled in 1.0.0 — see below)
- [x] Flip `console=True` → `console=False` in `OffAir.spec` for a
      release build (no console window on double-click) — done in
      1.0.1, with the stdout/stderr audit above
- [ ] Upload the setup exe as a GitHub Release asset
- [ ] Note the SmartScreen warning in the release notes
