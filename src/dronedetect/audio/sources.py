from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from ..errors import SourceError

SAMPLE_RATE = 16000
CHUNK = 16000
HOP = 8000


@dataclass
class Device:
    id: int
    name: str


class AudioSource(Protocol):
    def chunks(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class Chunker:
    """Нарезает поток на окна по `size` отсчётов с шагом `hop`."""

    def __init__(self, size: int = CHUNK, hop: int = HOP):
        self.size = size
        self.hop = hop
        self._buf = np.zeros(0, np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, np.asarray(samples, np.float32).reshape(-1)])
        out = []
        while self._buf.size >= self.size:
            out.append(self._buf[: self.size].copy())
            self._buf = self._buf[self.hop :]
        return out


class FileAudioSource:
    def __init__(self, path: Path, realtime: bool = True):
        import soundfile as sf
        from scipy.signal import resample_poly

        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            raise SourceError("Не удалось прочитать аудиофайл") from e
        if data.shape[0] == 0 or sr <= 0:
            raise SourceError("Не удалось прочитать аудиофайл")
        mono = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            frac = Fraction(SAMPLE_RATE, int(sr)).limit_denominator(1000)
            mono = resample_poly(mono, frac.numerator, frac.denominator).astype(np.float32)
        self._wave = mono.astype(np.float32)
        self._realtime = realtime
        self._stop = threading.Event()

    def chunks(self) -> Iterator[np.ndarray]:
        parts = Chunker().push(self._wave)
        if not parts:
            padded = np.zeros(CHUNK, np.float32)
            padded[: self._wave.size] = self._wave
            parts = [padded]
        for i, part in enumerate(parts):
            if self._stop.is_set():
                return
            if self._realtime and i > 0 and self._stop.wait(HOP / SAMPLE_RATE):
                return
            yield part

    def close(self) -> None:
        self._stop.set()


def list_microphones() -> list[Device]:
    import sounddevice as sd

    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        return []
    return [
        Device(i, str(d["name"]))
        for i, d in enumerate(devices)
        if d.get("max_input_channels", 0) > 0
    ]


class MicAudioSource:
    def __init__(self, device_id: int | None):
        import sounddevice as sd

        if not list_microphones():
            raise SourceError("Микрофон не найден")
        self._queue: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_id,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:  # noqa: BLE001
            raise SourceError("Не удалось открыть микрофон") from e

    def _callback(self, indata, frames, time_info, status) -> None:
        self._queue.put(indata[:, 0].copy())

    def chunks(self) -> Iterator[np.ndarray]:
        chunker = Chunker()
        while not self._closed.is_set():
            try:
                samples = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            yield from chunker.push(samples)

    def close(self) -> None:
        self._closed.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass
