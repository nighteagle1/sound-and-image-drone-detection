from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class AppConfig:
    audio_model: Path
    vision_model: Path
    audio_threshold: float = 0.5
    vision_threshold: float = 0.25
    port: int = 8000


def load_config(path: Path | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else base / p

    kwargs = {}
    for key, cast in (("audio_threshold", float), ("vision_threshold", float), ("port", int)):
        if key in data:
            kwargs[key] = cast(data[key])
    return AppConfig(
        audio_model=resolve(data.get("audio_model", "models/drone_audio.onnx")),
        vision_model=resolve(data.get("vision_model", "models/drone_yolov5.onnx")),
        **kwargs,
    )
