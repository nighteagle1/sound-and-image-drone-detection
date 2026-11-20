from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..errors import ModelNotFoundError

DEFAULT_SIZE = 640
PAD_VALUE = 114


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    out = cv2.copyMakeBorder(
        resized, top, size - nh - top, left, size - nw - left,
        cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3,
    )
    return out, ratio, (dw, dh)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (area_i + area_r - inter + 1e-9) <= iou]
    return keep


def postprocess(output, ratio, pad, conf, iou: float = 0.45) -> list[Detection]:
    pred = np.asarray(output)[0]
    scores = pred[:, 4] * pred[:, 5:].max(axis=1)
    mask = scores >= conf
    pred, scores = pred[mask], scores[mask]
    if not len(pred):
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = _nms(boxes, scores, iou)
    dw, dh = pad
    result = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        result.append(Detection(
            float((x1 - dw) / ratio), float((y1 - dh) / ratio),
            float((x2 - dw) / ratio), float((y2 - dh) / ratio), float(scores[i]),
        ))
    return result


class YoloDetector:
    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(f"Файл модели изображения не найден: {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        shape = inp.shape
        self._size = shape[2] if isinstance(shape[2], int) else DEFAULT_SIZE

    def detect(self, frame_bgr: np.ndarray, conf: float) -> list[Detection]:
        img, ratio, pad = letterbox(frame_bgr, self._size)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        out = self._session.run(None, {self._input: x})[0]
        return postprocess(out, ratio, pad, conf)

    def draw(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame_bgr.copy()
        for d in detections:
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
            cv2.putText(out, f"{d.score:.2f}", (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return out

    def close(self) -> None:
        self._session = None
