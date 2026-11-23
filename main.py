#!/usr/bin/env python3
"""OffAir — entry point (Phases 1–5).

Usage:
    python main.py [--output-dir DIR] [--no-gui] [--list-devices]

``--no-gui`` is a smoke check: it prints the configuration, lists the
audio devices it can see (empty on non-Windows, by design), and exits
without opening a window. Settings are loaded from
``~/.offair/settings.json`` when present (Phase 5); the
``--output-dir`` flag overrides the saved output folder for this run.
An uncaught exception is logged and shows the friendly crash dialog
(Phase 5) instead of silently exiting.

Windowed builds (PyInstaller ``console=False``, see
docs/PACKAGING.md): the frozen app has no console — ``sys.stdout``
and ``sys.stderr`` are None — so nothing in this module prints
unconditionally. The smoke check writes to the console when one is
attached and to ``offair.log`` when there is none; a bad command
line exits 2 the same way instead of crashing inside argparse.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.audio.devices import (
    find_teams_process,
    list_loopback_devices,
    list_microphones,
    pyaudiowpatch_status,
)
from app.config import (
    AppConfig,
    get_default_output_root,
    load_settings,
)
from app.logging_setup import install_exception_hook, setup_logging

logger = logging.getLogger("offair")


def _emit(text: str, *, error: bool = False) -> None:
    """Write one line to the console when there is one, else to the log.

    In a PyInstaller windowed build (``console=False``) ``sys.stdout``
    and ``sys.stderr`` are None and a bare ``print()`` raises
    AttributeError — so the smoke check and the PyQt6-missing error
    go through here: print when a stream exists, log to
    ``offair.log`` (wired by setup_logging() before this is called)
    when it does not.
    """
    stream = sys.stderr if error else sys.stdout
    if stream is not None:
        try:
            print(text, file=stream)
            return
        except (AttributeError, OSError, ValueError):
            pass  # Stream went away mid-write; fall through to the log.
    if error:
        logger.error("%s", text)
    else:
        logger.info("%s", text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offair",
        description="OffAir — local Teams call recorder, transcription, "
                    "diarization and speaker learning (CPU-only).")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Folder recordings are saved under "
             f"(default: {get_default_output_root()})")
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Smoke check only: print config/devices and exit, no window.")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List microphones and loopback devices, then exit.")
    parser.add_argument(
        "--mcp", action="store_true",
        help="Run the MCP server on stdio (see docs/MCP.md) "
             "instead of opening the window: `OffAir.exe --mcp`. "
             "No GUI, no network socket, local files only.")
    return parser


def smoke_check(config: AppConfig) -> int:
    """Print config + device status; return 0 when the app can start.

    Every line goes through :func:`_emit`: on a console the output
    is printed as before; in a windowed build (no console attached)
    the same lines land in ``offair.log`` instead — the frozen smoke
    test asserts on the exit code + the log file, never on stdout.
    """
    _emit("OffAir — CPU-only, local only, no GPU/CUDA")
    _emit(f"Output root : {config.output_root}")
    _emit(f"Sample rate : {config.sample_rate} Hz, "
          f"{config.channels} channel(s), chunk {config.chunk_frames} frames")
    _emit(f"Target app  : {config.app_name} {config.process_names}")
    _emit(f"WASAPI      : {pyaudiowpatch_status()}")
    mics = list_microphones()
    loops = list_loopback_devices()
    _emit(f"Microphones : {len(mics)} found")
    for mic in mics:
        _emit(f"  [{mic.index}] {mic.display_name()}")
    _emit(f"Loopback    : {len(loops)} found")
    for device in loops:
        _emit(f"  [{device.index}] {device.display_name()}")
    process = find_teams_process(config.process_names)
    _emit("Teams       : "
          + (f"running (pid {process['pid']}, {process['name']})"
             if process else "not running"))
    # Honest per-platform capture state (Track 3, app/platform_info.py):
    # on Linux this names sounddevice/PortAudio and the monitor source,
    # never the old Windows-only placeholder text.
    from app.platform_info import describe_capture_state
    info = describe_capture_state()
    _emit(f"Capture     : {info.backend} — {info.status}")
    if info.detail:
        _emit(f"              {info.detail}")
    if info.install_hint:
        _emit(info.install_hint)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        raise
    except (AttributeError, OSError, ValueError):
        # No console attached (windowed frozen build): argparse
        # prints usage to sys.stderr, which is None, so it raises
        # instead of exiting. Record the failure in the default log
        # file and exit 2 rather than crashing.
        setup_logging(get_default_output_root())
        logger.error("Could not parse command-line arguments %r: "
                     "no console to print usage to",
                     argv if argv is not None else sys.argv[1:])
        return 2
    # Phase 5: persisted settings first; --output-dir wins for this run.
    config = load_settings()
    if args.output_dir is not None:
        config.output_root = args.output_dir
    setup_logging(config.output_root)

    if args.mcp:
        # MCP mode: stdio server, no GUI, no crash dialog, no Qt.
        # setup_logging() above already wired the log file; the
        # console handler it adds writes to stderr, which is not
        # the MCP wire. --output-dir above is honoured.
        from app.mcp.server import run_mcp_server

        try:
            return run_mcp_server(config.output_root)
        except RuntimeError as exc:
            # Missing `mcp` package: plain stderr message, exit 2.
            _emit(f"offair --mcp: {exc}", error=True)
            return 2

    install_exception_hook(
        log_path=config.output_root / "offair.log",
        show_dialog=True)

    if args.list_devices or args.no_gui:
        return smoke_check(config)

    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        # No console to print to in a windowed build — _emit logs to
        # offair.log instead (the crash dialog needs Qt, so the log
        # file is the only channel here).
        _emit("PyQt6 is not installed — install it with "
              "`pip install -r requirements.txt`, or run with --no-gui "
              "for a smoke check.", error=True)
        return 2

    from app.resources import set_app_icon
    from app.ui.main_window import create_main_window
    from app.ui.theme import apply_theme_to_app

    app = QApplication(argv if argv is not None else sys.argv)
    # App icon (1.0.1): window/taskbar/dialogs show the OffAir mark.
    # Missing icon file is a warning, never a startup failure.
    set_app_icon(app)
    # Phase 6 fix: apply the theme at the QApplication level so EVERY
    # window (main window, Recording Browser, Settings dialog) follows
    # it — applying it to the main window alone left the other windows
    # in the default light style. The main window still applies it to
    # itself as well, so standalone construction stays themed.
    try:
        apply_theme_to_app(app, config.theme)
    except Exception:  # noqa: BLE001 — a bad theme must not block startup
        logging.getLogger("offair.ui").warning(
            "Could not apply theme %r", config.theme, exc_info=True)
    window = create_main_window(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
