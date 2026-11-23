# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for OffAir (Windows, CPU-only, no CUDA).

Build (on the Windows build machine, inside the venv):
    C:\\OffAir\\.venv\\Scripts\\pyinstaller.exe OffAir.spec

Output (onedir):  dist\\OffAir\\OffAir.exe  + support files
Run it from a plain cmd (no venv, no Python on PATH) to prove the
bundle is standalone. See docs/PACKAGING.md for the full story.

Why this spec looks the way it does
-----------------------------------
* The app lazy-imports its heavy modules inside functions
  (``from faster_whisper import WhisperModel`` in
  app/transcription/engine.py, sherpa_onnx / onnxruntime /
  speakeronnx / torch / pyannote in the diarization + embedding
  code). PyInstaller's static analysis does find function-level
  imports, but native DLLs and package data do not ride along on
  their own — hence ``collect_all`` for every native-heavy package.
* ``torch`` / ``pyannote.audio`` / ``speechbrain`` are EXCLUDED on
  purpose. They are only needed for the opt-in "Full" diarization
  mode and the SpeechBrain embedding backend; the app's default
  path (sherpa diarization + ONNX ECAPA embeddings + simple mode)
  never touches them, and the app raises a guided "not installed"
  error if a user picks those modes. Bundling torch would add
  ~1.5 GB for features that are off by default. Every torch /
  pyannote / speechbrain import in the code is inside a
  try/except ImportError with that guided error, so excluding them
  is safe — verified by the frozen-app smoke tests (see
  docs/PACKAGING.md).
* Models are NOT bundled: Whisper small (~466 MB) and the sherpa
  models (~33 MB) download once on first use into
  %USERPROFILE%\\.offair\\models, exactly like the pip install.
  The speakeronnx ECAPA weights (~20 MB) download the same way.
* Two data entries are bundled: the application icon
  (``assets/offair.ico`` -> ``assets/`` in the bundle, resolved at
  runtime by app/resources.py) and the ``docs/`` Markdown folder
  (``docs`` -> ``docs/``, rendered by the in-app Help tab via
  app/resources.docs_dir()): the app keeps settings/recordings
  under %USERPROFILE% (``~/OffAir/recordings``,
  ``~/.offair/settings.json``), never next to the exe.
* The Visual C++ runtime: the native DLLs (onnxruntime,
  ctranslate2, torch) link against VCRUNTIME140.dll. PyInstaller
  bundles the runtime DLLs it finds on the build machine, so the
  build machine must have the VC++ Redistributable installed
  (vc_redist.x64.exe) — see docs/PACKAGING.md. The target machine
  needs nothing extra.
"""
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Native-heavy packages: datas + binaries + hiddenimports, all in.
_tmp_ret = collect_all("faster_whisper")
fw_datas, fw_binaries, fw_hidden = _tmp_ret
_tmp_ret = collect_all("ctranslate2")
ct2_datas, ct2_binaries, ct2_hidden = _tmp_ret
_tmp_ret = collect_all("av")
av_datas, av_binaries, av_hidden = _tmp_ret
_tmp_ret = collect_all("onnxruntime")
ort_datas, ort_binaries, ort_hidden = _tmp_ret
_tmp_ret = collect_all("sherpa_onnx")
sherpa_datas, sherpa_binaries, sherpa_hidden = _tmp_ret
_tmp_ret = collect_all("speakeronnx")
spk_datas, spk_binaries, spk_hidden = _tmp_ret
_tmp_ret = collect_all("soundfile")
sf_datas, sf_binaries, sf_hidden = _tmp_ret
_tmp_ret = collect_all("pyaudiowpatch")
pap_datas, pap_binaries, pap_hidden = _tmp_ret
_tmp_ret = collect_all("comtypes")
ct_datas, ct_binaries, ct_hidden = _tmp_ret
_tmp_ret = collect_all("pycaw")
pycaw_datas, pycaw_binaries, pycaw_hidden = _tmp_ret
# MCP server (Nov 10 2026): the official MCP Python SDK, for
# `OffAir.exe --mcp` (stdio, read-only). Pure Python plus the
# pydantic-core / cryptography binary wheels — collect_all pulls
# the package data and the SDK's own deps (anyio, pydantic,
# starlette, httpx2, uvicorn) along. The SDK is only imported when
# --mcp is passed, so a missing bundle here degrades to a guided
# error instead of breaking the GUI.
_tmp_ret = collect_all("mcp")
mcp_datas, mcp_binaries, mcp_hidden = _tmp_ret

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=(
        fw_binaries + ct2_binaries + av_binaries + ort_binaries
        + sherpa_binaries + spk_binaries + sf_binaries + pap_binaries
        + ct_binaries + pycaw_binaries + mcp_binaries
    ),
    datas=(
        fw_datas + ct2_datas + av_datas + ort_datas
        + sherpa_datas + spk_datas + sf_datas + pap_datas
        + ct_datas + pycaw_datas + mcp_datas
        # The application icon (assets/offair.ico), resolved at
        # runtime by app/resources.py — the exe icon below is the
        # file's icon in Explorer; this entry is what the running
        # app loads for its window/taskbar/dialog icons.
        + [("assets/offair.ico", "assets")]
        # The docs/ Markdown folder, rendered by the in-app Help tab
        # (app/ui/help_view.py, resolved via
        # app/resources.docs_dir()). A frozen install without it
        # degrades gracefully — the Help tab shows a plain message
        # per missing file — but shipping it is the point: the Help
        # tab is the single source of truth rendered from these
        # files, so they must ride along.
        + [("docs", "docs")]
    ),
    hiddenimports=(
        [
            # Lazy imports the static analysis can miss or that live
            # behind try/except ImportError guards.
            "faster_whisper",
            "faster_whisper.transcribe",
            "faster_whisper.audio",
            "faster_whisper.vad",
            "faster_whisper.tokenizer",
            "ctranslate2",
            "av",
            "av.audio",
            "av.container",
            "onnxruntime",
            "sherpa_onnx",
            "speakeronnx",
            "soundfile",
            "numpy",
            "psutil",
            "comtypes",
            "comtypes.client",
            "comtypes.gen",
            "pyaudiowpatch",
            "pycaw",
            "pycaw.pycaw",
            "PyQt6.QtMultimedia",
            "PyQt6.QtWidgets",
            "PyQt6.QtCore",
            "PyQt6.QtGui",
            # huggingface_hub is used by faster-whisper's model download.
            "huggingface_hub",
            # certifi's CA bundle is used by the sherpa model downloader
            # (app/diarization/sherpa.py _ssl_context) on machines whose
            # system cert store is stale; it is collected anyway via
            # huggingface_hub, this just makes it explicit.
            "certifi",
            # MCP server (Nov 10 2026): the SDK's submodules are
            # imported lazily from app/mcp/server.py, behind a
            # try/except — list them explicitly so --mcp works in
            # the frozen build (collect_all("mcp") above covers the
            # datas/binaries; these cover the analysis).
            "mcp",
            "mcp.server",
            "mcp.server.mcpserver",
            "mcp.server.stdio",
            "mcp.types",
            "mcp.shared",
        ]
        + fw_hidden + ct2_hidden + av_hidden + ort_hidden
        + sherpa_hidden + spk_hidden + sf_hidden + pap_hidden
        + ct_hidden + pycaw_hidden + mcp_hidden
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Opt-in-only heavy stack (see header): the app degrades with
        # a guided error when these are absent. Excluding them keeps
        # the bundle from ballooning by ~1.5 GB.
        "torch",
        "torchaudio",
        "torchvision",
        "lightning",
        "pyannote",
        "pyannote.audio",
        "speechbrain",
        "scipy",
        "sklearn",
        "matplotlib",
        "pandas",
        "pytest",
        "tkinter",
        # Linux-only capture backend: not installed on Windows, and
        # the import is guarded in app/audio/devices.py.
        "sounddevice",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OffAir",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=False since 1.0.1: double-clicking the exe shows only
    # the app window, no console. The app was built for this — see
    # docs/PACKAGING.md "No console window (1.0.1)": logging goes to
    # offair.log only (setup_logging attaches no console handler
    # when sys.stderr is None), the --no-gui smoke check writes to
    # the log file when no console is attached, and uncaught errors
    # surface through the crash dialog, never a bare print.
    console=False,
    # The exe's own icon in Explorer / the taskbar: the multi-size
    # .ico in assets/ (source: assets/offair-icon.png, regenerated
    # with `python3 assets/make_icon.py`). Path is relative to this
    # spec file; the build runs from the repo root.
    icon="assets/offair.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OffAir",
)
