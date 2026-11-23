"""Phase 5 tests — settings persistence (app.config).

Settings live in JSON; a Hugging Face token is NEVER stored — only
the boolean ``hf_token_configured`` flag. A corrupt / partial /
unknown settings file must never break startup.
"""
from __future__ import annotations

import json

import pytest

from app.config import (
    AppConfig,
    config_from_settings,
    default_settings_path,
    load_settings,
    save_settings,
    settings_to_dict,
)


def test_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    config = AppConfig(
        output_root=tmp_path / "recordings", whisper_model="base",
        diarization_mode="full", hf_token_configured=True,
        near_live_enabled=True, nearlive_chunk_seconds=20.0,
        theme="light", ollama_enabled=True, ollama_model="llama3.1",
        ollama_base_url="http://localhost:11434")
    saved = save_settings(config, path)
    assert saved == path
    loaded = load_settings(path)
    assert loaded.output_root == config.output_root
    assert loaded.whisper_model == "base"
    assert loaded.diarization_mode == "full"
    assert loaded.hf_token_configured is True
    assert loaded.near_live_enabled is True
    assert loaded.nearlive_chunk_seconds == 20.0
    assert loaded.theme == "light"
    assert loaded.ollama_enabled is True


def test_settings_file_never_contains_a_token_value(tmp_path):
    """The saved file holds a boolean flag, never a token string."""
    path = tmp_path / "settings.json"
    save_settings(AppConfig(hf_token_configured=True), path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["hf_token_configured"] is True
    # No key may look like a token store, and no hf_ secret value exists.
    assert "hf_token" not in data
    assert "token" not in [k for k in data if k != "hf_token_configured"]
    assert "hf_" not in raw.replace("hf_token_configured", "")


def test_load_settings_missing_file_returns_defaults(tmp_path):
    loaded = load_settings(tmp_path / "no-such-file.json")
    assert loaded == AppConfig()


def test_fresh_install_defaults_nov_2026(tmp_path):
    """New defaults: sherpa diarization, hybrid search, auto-process.

    Whisper is now "base" (Nov 19 2026 — was "small"). A saved
    explicit choice must survive (see
    test_saved_explicit_choices_are_kept and
    test_saved_explicit_small_is_kept below) — only the defaults
    for fresh installs / missing keys changed.
    """
    from app.config import WHISPER_DEFAULT_MODEL

    assert WHISPER_DEFAULT_MODEL == "base"
    fresh = AppConfig()
    assert fresh.whisper_model == "base"
    assert fresh.diarization_mode == "sherpa"
    assert fresh.search_mode == "hybrid"
    assert fresh.auto_process_recordings is True
    # Ollama default (Nov 8 2026 research): qwen3.5:9b, not
    # llama3.1 — and the exports module must agree with config
    # (single source of truth, no drift).
    assert fresh.ollama_model == "qwen3.5:9b"
    from app.exports.local_llm import DEFAULT_OLLAMA_MODEL
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"
    loaded = load_settings(tmp_path / "no-such-file.json")
    assert loaded.diarization_mode == "sherpa"
    assert loaded.search_mode == "hybrid"
    assert loaded.auto_process_recordings is True
    assert loaded.ollama_model == "qwen3.5:9b"


def test_saved_explicit_choices_are_kept(tmp_path):
    """An existing user's saved "simple"/"text" is NOT rewritten."""
    path = tmp_path / "settings.json"
    saved = AppConfig(diarization_mode="simple", search_mode="text",
                      auto_process_recordings=False)
    save_settings(saved, path)
    loaded = load_settings(path)
    assert loaded.diarization_mode == "simple"
    assert loaded.search_mode == "text"
    assert loaded.auto_process_recordings is False


def test_saved_explicit_small_is_kept(tmp_path):
    """An EXPLICITLY picked "small" keeps small (Nov 20 2026, Item 8).

    This supersedes the Nov 19 2026 reading ("a saved small is
    always kept"): a stored "small" with NO explicit flag is the
    1.0.2 default, not a pick, and migrates to the current default
    (base) — see test_upgrade_pre_flag_small_migrates_to_base.
    Only a saved flag (or a stored tiny/medium, which no past
    default ever was) proves an explicit pick.
    """
    path = tmp_path / "settings.json"
    save_settings(AppConfig(whisper_model="small",
                            whisper_model_explicit=True), path)
    loaded = load_settings(path)
    assert loaded.whisper_model == "small"
    assert loaded.whisper_model_explicit is True
    assert loaded.effective_whisper_model == "small"
    # Same via a raw settings dict with the flag present.
    loaded2 = config_from_settings({
        "whisper_model": "small", "whisper_model_explicit": True})
    assert loaded2.effective_whisper_model == "small"


def test_upgrade_pre_flag_small_migrates_to_base(tmp_path):
    """Upgrade (Item 8): pre-flag "small" is the 1.0.2 default, so
    the current default (base) wins on load."""
    loaded = config_from_settings({"whisper_model": "small"})
    assert loaded.whisper_model_explicit is False
    assert loaded.effective_whisper_model == "base"
    # Same from a real pre-flag file on disk.
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"whisper_model": "small"}),
                    encoding="utf-8")
    assert load_settings(path).effective_whisper_model == "base"


def test_upgrade_pre_flag_tiny_and_medium_are_kept(tmp_path):
    """Upgrade (Item 8): no past default was ever tiny/medium, so
    a stored tiny/medium could only have been picked — kept."""
    for model in ("tiny", "medium"):
        loaded = config_from_settings({"whisper_model": model})
        assert loaded.whisper_model_explicit is True
        assert loaded.effective_whisper_model == model


def test_upgrade_pre_flag_base_or_missing_is_not_explicit():
    """Upgrade (Item 8): stored "base", missing, or invalid model
    with no flag → non-explicit, effective model base either way."""
    assert config_from_settings(
        {"whisper_model": "base"}).effective_whisper_model == "base"
    assert config_from_settings(
        {"whisper_model": "base"}).whisper_model_explicit is False
    assert config_from_settings({}).effective_whisper_model == "base"
    assert config_from_settings({}).whisper_model_explicit is False
    assert config_from_settings(
        {"whisper_model": "huge"}).effective_whisper_model == "base"
    assert config_from_settings(
        {"whisper_model": "huge"}).whisper_model_explicit is False


def test_explicit_flag_round_trips(tmp_path):
    """The flag is written by save_settings and read back."""
    path = tmp_path / "settings.json"
    save_settings(AppConfig(whisper_model="tiny",
                            whisper_model_explicit=True), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["whisper_model_explicit"] is True
    assert load_settings(path).effective_whisper_model == "tiny"
    save_settings(AppConfig(whisper_model="base"), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["whisper_model_explicit"] is False


def test_explicit_choice_survives_simulated_version_change(
        tmp_path, monkeypatch):
    """Explicit "small" survives a new default; a non-explicit
    setting follows the new default (Item 8)."""
    import app.config as config_module

    path = tmp_path / "settings.json"
    save_settings(AppConfig(whisper_model="small",
                            whisper_model_explicit=True), path)
    monkeypatch.setattr(config_module, "WHISPER_DEFAULT_MODEL", "tiny")
    assert load_settings(path).effective_whisper_model == "small"
    save_settings(AppConfig(whisper_model="base"), path)
    assert load_settings(path).effective_whisper_model == "tiny"


def test_auto_process_recordings_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(AppConfig(auto_process_recordings=False), path)
    assert load_settings(path).auto_process_recordings is False
    save_settings(AppConfig(auto_process_recordings=True), path)
    assert load_settings(path).auto_process_recordings is True
    # A settings file written before the key existed defaults to on.
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["auto_process_recordings"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_settings(path).auto_process_recordings is True


def test_load_settings_corrupt_json_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_settings(path) == AppConfig()
    path.write_text('["not", "a", "dict"]', encoding="utf-8")
    assert load_settings(path) == AppConfig()


def test_config_from_settings_ignores_unknown_keys_and_bad_values():
    config = config_from_settings({
        "whisper_model": "base",
        "theme": "light",
        "unknown_future_key": "ignored",
        "hf_token": "hf_SHOULD_NEVER_BE_READ",
    })
    assert config.whisper_model == "base"
    assert config.theme == "light"
    # Bad values fall back to defaults, one key at a time.
    config2 = config_from_settings({
        "whisper_model": "huge",      # invalid model
        "theme": "neon",              # invalid theme
        "diarization_mode": "magic",  # invalid mode
        "nearlive_chunk_seconds": 999,
        "ollama_enabled": True,       # valid — kept
    })
    assert config2.whisper_model == "base"
    assert config2.theme == "light"
    assert config2.diarization_mode == "sherpa"
    assert config2.nearlive_chunk_seconds == 25.0
    assert config2.ollama_enabled is True


def test_app_config_validates_new_fields():
    with pytest.raises(ValueError, match="theme"):
        AppConfig(theme="neon")
    with pytest.raises(ValueError, match="diarization mode"):
        AppConfig(diarization_mode="magic")
    with pytest.raises(ValueError, match="nearlive_chunk_seconds"):
        AppConfig(nearlive_chunk_seconds=1.0)


def test_default_settings_path_is_under_home_dot_offair():
    path = default_settings_path()
    assert path.name == "settings.json"
    assert path.parent.name == ".offair"


def test_settings_to_dict_has_no_token_key():
    keys = set(settings_to_dict(AppConfig()))
    assert "hf_token" not in keys
    assert "token" not in keys


# ------------------------------------------------------- tinyfix default ---
def test_tinyfix_fresh_install_default_is_off():
    """Nov 22 2026: tinyfix flipped from default-ON to opt-in
    after test 2 showed the stage changing correct words on
    disfluent real speech (see
    docs/notes/cleanup-repeat-rule-2026-11-22.md). Mode default
    ("careful") is unchanged."""
    from app.config import TINYFIX_DEFAULT_ENABLED, TINYFIX_DEFAULT_MODE

    assert TINYFIX_DEFAULT_ENABLED is False
    assert TINYFIX_DEFAULT_MODE == "careful"
    fresh = AppConfig()
    assert fresh.tinyfix_enabled is False
    assert fresh.tinyfix_mode == "careful"
    assert fresh.tinyfix_enabled_explicit is False


def test_tinyfix_explicit_opt_in_is_kept(tmp_path):
    """A saved opt-in WITH the explicit flag survives loading."""
    path = tmp_path / "settings.json"
    save_settings(AppConfig(tinyfix_enabled=True,
                            tinyfix_enabled_explicit=True), path)
    loaded = load_settings(path)
    assert loaded.tinyfix_enabled is True
    assert loaded.tinyfix_enabled_explicit is True
    # Same via a raw settings dict with the flag present.
    loaded2 = config_from_settings({
        "tinyfix_enabled": True, "tinyfix_enabled_explicit": True})
    assert loaded2.tinyfix_enabled is True
    assert loaded2.tinyfix_enabled_explicit is True


def test_upgrade_pre_flag_tinyfix_true_migrates_to_off(tmp_path):
    """Upgrade: a pre-flag file with tinyfix_enabled=True cannot
    be told apart from the old ON default (every settings save
    wrote the checkbox state, checked by default), so it is
    NOT an explicit opt-in and the current default (OFF) wins
    — mirroring the whisper-model migration (Nov 20 2026)."""
    loaded = config_from_settings({"tinyfix_enabled": True})
    assert loaded.tinyfix_enabled is False
    assert loaded.tinyfix_enabled_explicit is False
    # Same from a real pre-flag file on disk.
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"tinyfix_enabled": True}),
                    encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.tinyfix_enabled is False
    assert loaded.tinyfix_enabled_explicit is False


def test_upgrade_pre_flag_tinyfix_false_or_missing_stays_off():
    """Stored False, or no key at all, with no flag → OFF,
    non-explicit."""
    assert config_from_settings(
        {"tinyfix_enabled": False}).tinyfix_enabled is False
    assert config_from_settings(
        {"tinyfix_enabled": False}).tinyfix_enabled_explicit is False
    assert config_from_settings({}).tinyfix_enabled is False
    assert config_from_settings({}).tinyfix_enabled_explicit is False


def test_tinyfix_explicit_flag_round_trips(tmp_path):
    """The flag is written by save_settings and read back."""
    path = tmp_path / "settings.json"
    save_settings(AppConfig(tinyfix_enabled=True,
                            tinyfix_enabled_explicit=True), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tinyfix_enabled_explicit"] is True
    assert load_settings(path).tinyfix_enabled is True
    save_settings(AppConfig(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tinyfix_enabled_explicit"] is False
    assert load_settings(path).tinyfix_enabled is False
