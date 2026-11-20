from pathlib import Path

from dronedetect.config import AppConfig, load_config


def test_load_config_resolves_relative_paths(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "audio_model: models/a.onnx\nvision_model: models/v.onnx\n", encoding="utf-8"
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg, AppConfig)
    assert cfg.audio_model == tmp_path / "models" / "a.onnx"
    assert cfg.vision_model == tmp_path / "models" / "v.onnx"
    assert cfg.audio_model.is_absolute()


def test_defaults_when_keys_missing(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("audio_model: a.onnx\nvision_model: v.onnx\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.audio_threshold == 0.5
    assert cfg.vision_threshold == 0.25
    assert cfg.port == 8000
