from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from ..audio.sources import Device
from ..errors import SourceError

_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class VideoSource(Protocol):
    def frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class CameraSource:
    def __init__(self, index: int):
        self._cap = cv2.VideoCapture(int(index), _BACKEND)
        if not self._cap.isOpened():
            self._cap.release()
            raise SourceError("Камера не найдена или занята")
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def frames(self) -> Iterator[np.ndarray]:
        while not self._stop.is_set():
            with self._lock:
                ok, frame = self._cap.read() if not self._stop.is_set() else (False, None)
            if not ok:
                return
            yield frame

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._cap.release()


def list_cameras(max_index: int = 4) -> list[Device]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _BACKEND)
        try:
            if cap.isOpened():
                found.append(Device(i, f"Камера {i}"))
        finally:
            cap.release()
    return found


class FileVideoSource:
    def __init__(self, path: Path, realtime: bool = True):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap.release()
            raise SourceError("Не удалось прочитать видеофайл")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._period = 1.0 / fps if fps and fps > 1 else 1.0 / 25
        self._realtime = realtime
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def frames(self) -> Iterator[np.ndarray]:
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                ok, frame = self._cap.read() if not self._stop.is_set() else (False, None)
            if not ok:
                return
            if self._realtime:
                delay = next_t - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    return
                next_t += self._period
            yield frame

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._cap.release()
