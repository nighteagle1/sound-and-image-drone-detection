from pathlib import Path

import numpy as np
import pytest

from dronedetect.errors import ModelNotFoundError
from dronedetect.vision.detector import Detection, YoloDetector, letterbox, postprocess

MODEL = Path(__file__).resolve().parents[1] / "models" / "drone_yolov5.onnx"


def _out(rows):
    """rows: (cx, cy, w, h, obj, cls) -> (1, N, 6)"""
    return np.array([rows], dtype=np.float32)


def test_letterbox_keeps_aspect():
    img, ratio, pad = letterbox(np.zeros((480, 640, 3), np.uint8), 640)
    assert img.shape == (640, 640, 3)
    assert ratio == 1.0
    assert pad == (0.0, 80.0)


def test_postprocess_filters_by_conf():
    out = _out([(100, 100, 40, 40, 0.9, 1.0), (400, 400, 40, 40, 0.1, 1.0)])
    dets = postprocess(out, 1.0, (0.0, 0.0), conf=0.25)
    assert len(dets) == 1 and isinstance(dets[0], Detection)
    assert dets[0].score == pytest.approx(0.9)


def test_postprocess_nms_merges_overlap():
    out = _out([(100, 100, 40, 40, 0.9, 1.0), (102, 101, 40, 40, 0.8, 1.0)])
    assert len(postprocess(out, 1.0, (0.0, 0.0), conf=0.25)) == 1


def test_postprocess_maps_back_to_original():
    out = _out([(320, 320, 100, 100, 0.9, 1.0)])
    d = postprocess(out, 1.0, (0.0, 80.0), conf=0.25)[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((270, 190, 370, 290))
    d = postprocess(out, 0.5, (0.0, 80.0), conf=0.25)[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((540, 380, 740, 580))


def test_detect_on_blank_frame_returns_list():
    det = YoloDetector(MODEL)
    res = det.detect(np.zeros((480, 640, 3), np.uint8), conf=0.25)
    assert isinstance(res, list)
    drawn = det.draw(np.zeros((480, 640, 3), np.uint8), [Detection(10, 10, 100, 100, 0.8)])
    assert drawn.shape == (480, 640, 3) and drawn.any()


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError) as e:
        YoloDetector(tmp_path / "x.onnx")
    assert "модел" in e.value.message
