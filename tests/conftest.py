"""Shared fixtures for the OffAir test suite.

The ``no_sd`` fixture makes sounddevice genuinely unimportable for one
test. The old per-file versions only deleted ``"sounddevice"`` from
``sys.modules`` — which does nothing once the package is installed on
disk (``requirements.txt`` lists ``sounddevice`` for Linux, so a
fully-installed Linux venv HAS it): the next ``import sounddevice``
just re-imported it from site-packages and the "without sounddevice"
tests failed. This version installs a meta-path finder that raises
``ImportError`` for ``sounddevice`` / ``_sounddevice``, so the
production ``load_sounddevice()`` try/except sees exactly the
missing-dependency case.
"""
import importlib.abc
import sys

import pytest


class BlockSounddevice(importlib.abc.MetaPathFinder):
    """Meta-path finder that makes sounddevice unimportable."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("sounddevice", "_sounddevice") or fullname.startswith(
            "sounddevice."
        ):
            raise ImportError("sounddevice is blocked for this test")
        return None


class BlockPyaudiowpatch(importlib.abc.MetaPathFinder):
    """Meta-path finder that makes pyaudiowpatch unimportable.

    Same problem as sounddevice (see the module docstring): deleting
    the name from ``sys.modules`` cannot block re-import on a machine
    where the package is installed — the Windows VM HAS pyaudiowpatch,
    so "missing pyaudiowpatch" tests must block the import itself.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("pyaudiowpatch", "_pyaudiowpatch") or fullname.startswith(
            "pyaudiowpatch."
        ):
            raise ImportError("pyaudiowpatch is blocked for this test")
        return None


@pytest.fixture
def no_sd(monkeypatch):
    """Make sure no sounddevice (real or fake) is importable."""
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.delitem(sys.modules, "_sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockSounddevice(), *sys.meta_path])
    return None


@pytest.fixture
def no_pyaudiowpatch(monkeypatch):
    """Make sure no pyaudiowpatch (real or fake) is importable."""
    monkeypatch.delitem(sys.modules, "pyaudiowpatch", raising=False)
    monkeypatch.delitem(sys.modules, "_pyaudiowpatch", raising=False)
    monkeypatch.setattr(
        sys, "meta_path", [BlockPyaudiowpatch(), *sys.meta_path])
    return None
