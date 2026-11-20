"""Лог-мел признаки на NumPy, совпадающие с torchaudio MelSpectrogram + AmplitudeToDB."""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
N_MELS = 64
F_MIN = 50.0
F_MAX = 5500.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank() -> np.ndarray:
    """(n_freqs, n_mels), HTK-шкала, без нормировки (как torchaudio по умолчанию)."""
    n_freqs = N_FFT // 2 + 1
    all_freqs = np.linspace(0, SAMPLE_RATE // 2, n_freqs)
    f_pts = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up))


_FB = _mel_filterbank()
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # периодический Hann


def log_mel(wave: np.ndarray) -> np.ndarray:
    """float32 моно 16 кГц, длина ≥ 1 -> float32 (1, 1, 64, T)."""
    x = np.asarray(wave, dtype=np.float64).reshape(-1)
    pad = N_FFT // 2
    if x.size <= pad:  # reflect-дополнение требует длину > N_FFT//2
        x = np.pad(x, (0, pad + 1 - x.size))
    x = np.pad(x, pad, mode="reflect")
    n_frames = 1 + (x.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _WINDOW, axis=1)) ** 2  # (T, 257)
    mel = spec @ _FB  # (T, 64)
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    out = ((db + 40.0) / 40.0).T
    return out[None, None].astype(np.float32)
