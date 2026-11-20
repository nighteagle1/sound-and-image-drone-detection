from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..errors import ModelNotFoundError
from .features import SAMPLE_RATE, log_mel


class AudioDetector:
    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(f"Файл звуковой модели не найден: {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def predict(self, chunk: np.ndarray) -> float:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (SAMPLE_RATE,):
            raise ValueError(f"Ожидался фрагмент из {SAMPLE_RATE} отсчётов, получено {chunk.shape}")
        logit = self._session.run(None, {self._input: log_mel(chunk)})[0].reshape(-1)[0]
        return float(1.0 / (1.0 + np.exp(-float(logit))))

    def close(self) -> None:
        self._session = None


class DetectionSmoother:
    """Включение после on_after окон подряд выше порога, выключение после off_after ниже."""

    def __init__(self, on_after: int = 2, off_after: int = 2):
        self.on_after = on_after
        self.off_after = off_after
        self.reset()

    def reset(self) -> None:
        self._state = False
        self._streak = 0

    def update(self, prob: float, threshold: float) -> bool:
        above = prob >= threshold
        if above != self._state:
            self._streak += 1
            if self._streak >= (self.on_after if above else self.off_after):
                self._state = above
                self._streak = 0
        else:
            self._streak = 0
        return self._state
