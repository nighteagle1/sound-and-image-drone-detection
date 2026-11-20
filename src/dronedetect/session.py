from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

import cv2

from .audio.detector import AudioDetector, DetectionSmoother
from .audio.sources import FileAudioSource, MicAudioSource
from .config import AppConfig
from .errors import DroneError
from .vision.detector import YoloDetector
from .vision.sources import CameraSource, FileVideoSource


class ConflictError(DroneError):
    pass


class EventHub:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class _Run:
    def __init__(self, mode, source, detector):
        self.mode = mode
        self.source = source
        self.detector = detector
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None


class DetectionSession:
    def __init__(self, config: AppConfig, hub: EventHub):
        self._config = config
        self._hub = hub
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self._threshold = config.audio_threshold
        self._jpeg: bytes | None = None
        self._frame_seq = 0

    # --- публичный API -------------------------------------------------
    def start(self, mode: Literal["audio", "vision"], source: dict, threshold: float | None = None) -> None:
        if mode not in ("audio", "vision"):
            raise DroneError("Неизвестный режим")
        with self._lock:
            if self._run is not None:
                raise ConflictError("Обнаружение уже запущено. Сначала нажмите «Стоп».")
            if threshold is not None:
                self._validate(threshold)
                self._threshold = threshold
            else:
                self._threshold = (
                    self._config.audio_threshold if mode == "audio" else self._config.vision_threshold
                )
            detector = (
                AudioDetector(self._config.audio_model)
                if mode == "audio"
                else YoloDetector(self._config.vision_model)
            )
            try:
                src = self._open_source(mode, source)
            except Exception:
                detector.close()
                raise
            run = _Run(mode, src, detector)
            self._jpeg = None
            run.thread = threading.Thread(target=self._worker, args=(run,), daemon=True)
            self._run = run
            run.thread.start()

    def stop(self) -> None:
        with self._lock:
            run = self._run
            if run is None:
                return
            run.stop_event.set()
            run.source.close()
        run.thread.join(timeout=5)
        with self._lock:
            if self._run is run:  # поток не успел завершиться
                self._run = None
                run.detector.close()

    def set_threshold(self, value: float) -> None:
        self._validate(value)
        self._threshold = float(value)

    def status(self) -> dict:
        run = self._run
        return {"running": run is not None, "mode": run.mode if run else None}

    def latest_jpeg(self) -> bytes | None:
        return self._jpeg

    def frame_seq(self) -> int:
        return self._frame_seq

    # --- внутреннее -----------------------------------------------------
    @staticmethod
    def _validate(value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise DroneError("Порог должен быть числом от 0 до 1")

    @staticmethod
    def _open_source(mode: str, source: dict):
        kind = source.get("kind")
        if kind == "file":
            path = Path(source["path"])
            return FileAudioSource(path) if mode == "audio" else FileVideoSource(path)
        if kind == "device":
            device_id = source.get("device_id")
            if mode == "audio":
                return MicAudioSource(device_id)
            return CameraSource(0 if device_id is None else device_id)
        raise DroneError("Не указан источник")

    def _worker(self, run: _Run) -> None:
        error: str | None = None
        try:
            if run.mode == "audio":
                self._audio_loop(run)
            else:
                self._vision_loop(run)
        except Exception as e:  # noqa: BLE001
            error = e.message if isinstance(e, DroneError) else f"Ошибка обработки: {e}"
        stopped = run.stop_event.is_set()
        self._finalize(run)
        if error is not None:
            self._hub.publish({"type": "error", "message": error})
        elif not stopped:
            self._hub.publish({"type": "ended"})

    def _finalize(self, run: _Run) -> None:
        with self._lock:
            if self._run is run:
                self._run = None
        try:
            run.source.close()
        finally:
            run.detector.close()

    def _audio_loop(self, run: _Run) -> None:
        smoother = DetectionSmoother()
        t0 = time.monotonic()
        for chunk in run.source.chunks():
            if run.stop_event.is_set():
                return
            prob = run.detector.predict(chunk)
            detected = smoother.update(prob, self._threshold)
            self._hub.publish({"type": "audio", "t": round(time.monotonic() - t0, 2),
                               "prob": prob, "detected": detected})

    def _vision_loop(self, run: _Run) -> None:
        stamps: deque[float] = deque(maxlen=30)
        for frame in run.source.frames():
            if run.stop_event.is_set():
                return
            dets = run.detector.detect(frame, self._threshold)
            ok, buf = cv2.imencode(".jpg", run.detector.draw(frame, dets), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._jpeg = buf.tobytes()
                self._frame_seq += 1
            now = time.monotonic()
            stamps.append(now)
            fps = (len(stamps) - 1) / (now - stamps[0]) if len(stamps) > 1 and now > stamps[0] else 0.0
            self._hub.publish({"type": "vision", "fps": round(fps, 1),
                               "count": len(dets), "detected": len(dets) > 0})
