import numpy as np
import pytest
import soundfile as sf

from dronedetect.audio.sources import (
    Chunker,
    FileAudioSource,
    MicAudioSource,
    list_microphones,
)
from dronedetect.errors import SourceError


def test_chunker_overlap():
    c = Chunker()
    assert len(c.push(np.zeros(16000, np.float32))) == 1
    assert len(c.push(np.zeros(8000, np.float32))) == 1
    assert len(c.push(np.zeros(4000, np.float32))) == 0


def test_file_source_stereo_44k_yields_mono_16k_chunks(tmp_path):
    p = tmp_path / "s.wav"
    sf.write(p, np.random.default_rng(0).normal(0, 0.1, (88200, 2)).astype(np.float32), 44100)
    src = FileAudioSource(p, realtime=False)
    chunks = list(src.chunks())
    src.close()
    assert len(chunks) >= 2
    for c in chunks:
        assert c.shape == (16000,) and c.dtype == np.float32


def test_short_file_single_padded_chunk(tmp_path):
    p = tmp_path / "short.wav"
    sf.write(p, np.full(4800, 0.1, np.float32), 16000)
    chunks = list(FileAudioSource(p, realtime=False).chunks())
    assert len(chunks) == 1
    assert chunks[0].shape == (16000,)
    assert chunks[0][-1] == 0.0 and chunks[0][0] != 0.0


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(SourceError):
        FileAudioSource(p, realtime=False)


def test_garbage_bytes_raises(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"this is not audio" * 100)
    with pytest.raises(SourceError) as e:
        FileAudioSource(p, realtime=False)
    assert "аудиофайл" in e.value.message


def test_mic_no_devices_raises(monkeypatch):
    import sounddevice

    monkeypatch.setattr(sounddevice, "query_devices", lambda *a, **k: [])
    with pytest.raises(SourceError) as e:
        MicAudioSource(None)
    assert "Микрофон не найден" in e.value.message
    assert list_microphones() == []
