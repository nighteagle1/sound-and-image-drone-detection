from pathlib import Path

import numpy as np

from dronedetect.audio.features import log_mel

GOLDEN = Path(__file__).parent / "data" / "audio_golden.npz"


def test_shape_for_one_second():
    out = log_mel(np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32))
    assert out.shape == (1, 1, 64, 101)
    assert out.dtype == np.float32


def test_silence_is_finite_and_floor():
    out = log_mel(np.zeros(16000, np.float32))
    assert np.isfinite(out).all()
    assert np.allclose(out, -1.5)


def test_very_short_input_is_finite():
    out = log_mel(np.ones(10, np.float32))
    assert out.shape[:3] == (1, 1, 64)
    assert np.isfinite(out).all()


def test_matches_torchaudio_golden():
    golden = np.load(GOLDEN)
    assert np.allclose(log_mel(golden["wave"]), golden["log_mel"], atol=1e-3, rtol=1e-3)
