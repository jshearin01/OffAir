"""Packaging guards for the 1.0.1 release — no console window, real icon.

Two halves, like test_ui_phase6.py:

* Qt-free (run in the plain test env): the spec really says
  ``console=False`` and wires the .ico, the installer points at the
  .ico, the icon files exist with all four sizes, setup_logging()
  writes to the log file with no console handler when there is no
  console (``sys.stderr`` is None, as in a PyInstaller windowed
  build), ``main(["--no-gui"])`` exits 0 and logs instead of
  printing when the streams are None, and the startup path has no
  unconditional ``print()``.
* Qt-gated (``pytest.importorskip("PyQt6")`` per test — run with the
  gui-shots venv, offscreen): constructing the main window prints
  nothing, and the window + QApplication carry the OffAir icon.

Run the Qt half with::

    QT_QPA_PLATFORM=offscreen \\
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \\
      -m pytest tests/test_packaging.py -q
"""
from __future__ import annotations

import ast
import logging
import os
import struct
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "OffAir.spec"
ISS = REPO / "OffAir.iss"
ICO = REPO / "assets" / "offair.ico"
PNG = REPO / "assets" / "offair-icon.png"


# ------------------------------------------------------------ Qt-free -------
def test_spec_has_console_false_and_wires_the_icon():
    """Regression guard: the frozen app must launch with no console.

    Reads the spec text — a real guard, not a mock. The spec is the
    thing PyInstaller executes, so asserting on its text is asserting
    on the build itself.
    """
    spec = SPEC.read_text(encoding="utf-8")
    assert "console=False" in spec
    assert "console=True" not in spec
    assert 'icon="assets/offair.ico"' in spec
    assert '("assets/offair.ico", "assets")' in spec


def test_installer_setup_icon_points_at_the_ico():
    iss = ISS.read_text(encoding="utf-8")
    assert "SetupIconFile=assets\\offair.ico" in iss


def _ico_sizes(path: Path) -> set[tuple[int, int]]:
    """Sizes embedded in a .ico, parsed from its header (no Pillow)."""
    data = path.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert reserved == 0 and kind == 1, "not a .ico file"
    sizes = set()
    for i in range(count):
        width, height = struct.unpack_from("<BB", data, 6 + i * 16)
        sizes.add((width or 256, height or 256))
    return sizes


def _png_size(path: Path) -> tuple[int, int]:
    """Width/height of a PNG, parsed from its IHDR chunk (no Pillow)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    width, height = struct.unpack_from(">II", data, 16)
    return width, height


def test_icon_files_exist_with_all_four_sizes():
    assert PNG.is_file(), "master source assets/offair-icon.png missing"
    assert _png_size(PNG) == (1024, 1024)
    assert ICO.is_file(), "assets/offair.ico missing"
    assert _ico_sizes(ICO) == {(16, 16), (32, 32), (48, 48), (256, 256)}


def _clear_offair_handlers() -> None:
    logger = logging.getLogger("offair")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_setup_logging_without_stderr_is_file_only(tmp_path, monkeypatch):
    """Windowed build: sys.stderr is None, so no console handler.

    setup_logging() must still write to the log file — the file is
    the only sink, and the crash dialog points the user at it.
    """
    from app.logging_setup import setup_logging

    monkeypatch.setattr(sys, "stderr", None)
    _clear_offair_handlers()
    try:
        logger = setup_logging(tmp_path)
        assert not any(isinstance(h, logging.StreamHandler)
                       and not isinstance(h, RotatingFileHandler)
                       for h in logger.handlers)
        rotating = [h for h in logger.handlers
                    if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        logger.info("file-only probe")
        for handler in logger.handlers:
            handler.flush()
        assert "file-only probe" in (tmp_path / "offair.log").read_text(
            encoding="utf-8")
    finally:
        _clear_offair_handlers()


def test_setup_logging_with_stderr_keeps_console_handler(tmp_path):
    """With a console (dev runs), the console handler is still added."""
    from app.logging_setup import setup_logging

    assert sys.stderr is not None  # the test env has a console
    _clear_offair_handlers()
    try:
        logger = setup_logging(tmp_path)
        assert any(isinstance(h, logging.StreamHandler)
                   and not isinstance(h, RotatingFileHandler)
                   for h in logger.handlers)
    finally:
        _clear_offair_handlers()


def test_main_no_gui_without_console_exits_zero_and_logs(tmp_path,
                                                         monkeypatch):
    """The frozen smoke check with no console: exit 0, log, no crash.

    Simulates a PyInstaller windowed build (``console=False``), where
    sys.stdout/sys.stderr are None and a bare print() raises
    AttributeError. main() must return 0 and the smoke lines must
    land in offair.log instead of stdout.
    """
    from main import main

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    previous_hook = sys.excepthook
    _clear_offair_handlers()
    try:
        assert main(["--no-gui", "--output-dir", str(tmp_path)]) == 0
        log = (tmp_path / "offair.log").read_text(encoding="utf-8")
        assert "Output root" in log
        assert "Capture" in log
    finally:
        sys.excepthook = previous_hook
        _clear_offair_handlers()


def test_main_py_prints_only_through_emit():
    """Source guard: main.py has no unconditional print().

    Every print() call in main.py must live inside _emit(), which
    prints when a console exists and logs to offair.log when it
    does not. A bare print anywhere else would crash a windowed
    build at startup.
    """
    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            enclosing = node
            while enclosing is not None and not isinstance(
                    enclosing, ast.FunctionDef):
                enclosing = parent.get(enclosing)
            name = enclosing.name if enclosing is not None else None
            assert name == "_emit", (
                f"print() outside _emit in main.py (in {name}) — "
                "a windowed build has no stdout to print to")


def test_startup_modules_have_no_print_calls():
    """No print() in any module imported on the startup path."""
    for relative in (
            "app/logging_setup.py",
            "app/config.py",
            "app/audio/devices.py",
            "app/platform_info.py",
            "app/resources.py",
    ):
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        prints = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)
                  and node.func.id == "print"]
        assert not prints, f"{relative} contains print() call(s)"


def test_app_icon_resolves_in_source_tree():
    from app.resources import app_icon_path, resource_path

    assert resource_path("assets/offair.ico") == ICO
    assert app_icon_path() == ICO


def test_app_icon_path_is_none_when_file_missing(monkeypatch):
    import app.resources as resources

    monkeypatch.setattr(resources, "APP_ICON_RELATIVE",
                        "assets/does-not-exist.ico")
    assert resources.app_icon_path() is None


# ------------------------------------------------------------ Qt-gated ------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/offair-demo/runtime")
    Path("/tmp/offair-demo/runtime").mkdir(parents=True, exist_ok=True)
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_main_window_construction_prints_nothing(qapp, capsys):
    """Startup must not print: a windowed build has nowhere to print to."""
    from app.ui.main_window import create_main_window

    window = create_main_window()
    assert window is not None
    out, err = capsys.readouterr()
    assert out == "", f"main-window construction printed to stdout: {out!r}"
    assert err == "", f"main-window construction printed to stderr: {err!r}"


def test_main_window_and_app_carry_the_offair_icon(qapp):
    """The window and the QApplication both show the OffAir mark."""
    from PyQt6.QtWidgets import QApplication

    from app.resources import set_app_icon
    from app.ui.main_window import create_main_window

    app = QApplication.instance()
    assert app is not None
    assert set_app_icon(app) is True
    assert not app.windowIcon().isNull()
    window = create_main_window()
    assert not window.windowIcon().isNull()
    sizes = {(s.width(), s.height())
             for s in window.windowIcon().availableSizes()}
    assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= sizes


def test_set_app_icon_missing_file_returns_false_without_crash(
        qapp, monkeypatch):
    """A missing icon file is a warning, never a startup failure."""
    import app.resources as resources
    from PyQt6.QtWidgets import QApplication

    monkeypatch.setattr(resources, "APP_ICON_RELATIVE",
                        "assets/does-not-exist.ico")
    app = QApplication.instance()
    assert app is not None
    # Returns False, does not raise, and does not touch the app icon.
    before = app.windowIcon()
    assert resources.set_app_icon(app) is False
    assert app.windowIcon().cacheKey() == before.cacheKey()
