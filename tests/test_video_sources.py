import cv2
import numpy as np
import pytest

from dronedetect.errors import SourceError
from dronedetect.vision import sources
from dronedetect.vision.sources import CameraSource, FileVideoSource, list_cameras


def test_file_source_yields_all_frames(tmp_path):
    p = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()
    src = FileVideoSource(p, realtime=False)
    frames = list(src.frames())
    src.close()
    assert len(frames) == 10
    assert all(f.shape == (48, 64, 3) for f in frames)


def test_garbage_file_raises(tmp_path):
    p = tmp_path / "bad.avi"
    p.write_bytes(b"not a video" * 100)
    with pytest.raises(SourceError) as e:
        FileVideoSource(p, realtime=False)
    assert "видеофайл" in e.value.message


class _ClosedCap:
    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return False

    def release(self):
        pass


class _OpenOnly:
    def __init__(self, index, *a, **k):
        self.index = index

    def isOpened(self):
        return self.index in (0, 2)

    def release(self):
        pass


def test_camera_missing_raises(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _ClosedCap)
    with pytest.raises(SourceError) as e:
        CameraSource(0)
    assert "Камера не найдена или занята" in e.value.message


def test_list_cameras_returns_only_opened(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _OpenOnly)
    assert [d.id for d in list_cameras()] == [0, 2]
