from pathlib import Path

import numpy as np
import pytest

from dronedetect.audio.detector import AudioDetector, DetectionSmoother
from dronedetect.errors import ModelNotFoundError

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "drone_audio.onnx"
GOLDEN = np.load(ROOT / "tests" / "data" / "audio_golden.npz")


@pytest.fixture(scope="module")
def det():
    d = AudioDetector(MODEL)
    yield d
    d.close()


def test_predict_matches_pytorch_golden(det):
    assert abs(det.predict(GOLDEN["wave"]) - float(GOLDEN["prob"])) < 1e-3


def test_predict_silence_in_range(det):
    p = det.predict(np.zeros(16000, np.float32))
    assert 0.0 <= p <= 1.0


def test_predict_wrong_length_raises(det):
    with pytest.raises(ValueError):
        det.predict(np.zeros(100, np.float32))


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError) as e:
        AudioDetector(tmp_path / "nope.onnx")
    assert "модел" in e.value.message


def test_smoother_turns_on_after_two():
    s = DetectionSmoother()
    assert s.update(0.9, 0.5) is False
    assert s.update(0.9, 0.5) is True


def test_smoother_turns_off_after_two():
    s = DetectionSmoother()
    s.update(0.9, 0.5), s.update(0.9, 0.5)
    assert s.update(0.1, 0.5) is True
    assert s.update(0.1, 0.5) is False


def test_smoother_single_dip_ignored():
    s = DetectionSmoother()
    s.update(0.9, 0.5), s.update(0.9, 0.5)
    assert s.update(0.1, 0.5) is True
    assert s.update(0.9, 0.5) is True
    assert s.update(0.1, 0.5) is True
