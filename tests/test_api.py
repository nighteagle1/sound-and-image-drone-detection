import queue
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from dronedetect import server
from dronedetect.config import AppConfig
from dronedetect.server import create_app

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = np.load(ROOT / "tests" / "data" / "audio_golden.npz")


def _cfg(tmp_path, real=True):
    if real:
        return AppConfig(ROOT / "models" / "drone_audio.onnx", ROOT / "models" / "drone_yolov5.onnx")
    return AppConfig(tmp_path / "no_a.onnx", tmp_path / "no_v.onnx")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(_cfg(tmp_path))) as c:
        yield c
        c.post("/api/stop")


@pytest.fixture
def client_nomodels(tmp_path):
    with TestClient(create_app(_cfg(tmp_path, real=False))) as c:
        yield c


def _upload(client, path: Path, name=None):
    with open(path, "rb") as f:
        r = client.post("/api/upload", files={"file": (name or path.name, f)})
    assert r.status_code == 200, r.text
    return r.json()["file_id"]


def _long_wav(tmp_path, seconds=30):
    p = tmp_path / "long.wav"
    sf.write(p, np.random.default_rng(0).normal(0, 0.05, 16000 * seconds).astype(np.float32), 16000)
    return p


def _start_audio(client, file_id):
    return client.post("/api/start", json={"mode": "audio", "source": {"kind": "file", "file_id": file_id}})


def test_status_reports_missing_models(client_nomodels):
    r = client_nomodels.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["audio_available"] is False and body["vision_available"] is False
    assert body["running"] is False


def test_status_reports_available_models(client):
    body = client.get("/api/status").json()
    assert body["audio_available"] is True and body["vision_available"] is True


def test_start_without_model_returns_400_with_text(client_nomodels):
    r = client_nomodels.post("/api/start", json={"mode": "audio", "source": {"kind": "device"}})
    assert r.status_code == 400
    assert "модел" in r.json()["error"]
    assert client_nomodels.get("/api/status").status_code == 200


def test_start_invalid_body_returns_400_russian(client):
    r = client.post("/api/start", json={"mode": "bogus"})
    assert r.status_code == 400 and r.json()["error"]


def test_start_audio_with_golden_wav_file(client, tmp_path):
    p = tmp_path / "golden.wav"
    sf.write(p, GOLDEN["wave"], 16000)
    hub_q = client.app.state.hub.subscribe()
    r = _start_audio(client, _upload(client, p))
    assert r.status_code == 200, r.text
    event = hub_q.get(timeout=10)
    assert event["type"] == "audio" and 0.0 <= event["prob"] <= 1.0
    assert event["prob"] > 0.5
    ended = hub_q.get(timeout=10)
    assert ended == {"type": "ended"}
    assert client.post("/api/stop").status_code == 200
    assert client.get("/api/status").json()["running"] is False


def test_start_twice_returns_409(client, tmp_path):
    fid = _upload(client, _long_wav(tmp_path))
    assert _start_audio(client, fid).status_code == 200
    r = _start_audio(client, fid)
    assert r.status_code == 409 and r.json()["error"]


def test_stop_when_idle_is_ok(client):
    r = client.post("/api/stop")
    assert r.status_code == 200 and r.json()["running"] is False


def test_start_other_mode_while_running_returns_409(client, tmp_path):
    assert _start_audio(client, _upload(client, _long_wav(tmp_path))).status_code == 200
    r = client.post("/api/start", json={"mode": "vision", "source": {"kind": "device", "device_id": 0}})
    assert r.status_code == 409
    assert client.get("/api/status").json()["mode"] == "audio"


def test_upload_garbage_then_start_returns_400(client, tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"garbage" * 1000)
    r = _start_audio(client, _upload(client, p))
    assert r.status_code == 400 and "аудиофайл" in r.json()["error"]
    assert client.get("/api/status").json()["running"] is False


def test_start_unknown_file_id_returns_400(client):
    r = _start_audio(client, "does-not-exist")
    assert r.status_code == 400


def test_upload_too_large_returns_413(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1000)
    p = tmp_path / "big.wav"
    p.write_bytes(b"x" * 5000)
    with open(p, "rb") as f:
        r = client.post("/api/upload", files={"file": ("big.wav", f)})
    assert r.status_code == 413 and r.json()["error"]


def test_vision_with_video_file_publishes_events_and_jpeg(client, tmp_path):
    p = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(30):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    hub_q = client.app.state.hub.subscribe()
    r = client.post("/api/start", json={"mode": "vision",
                                         "source": {"kind": "file", "file_id": _upload(client, p)}})
    assert r.status_code == 200, r.text
    event = hub_q.get(timeout=10)
    assert event["type"] == "vision" and event["count"] == 0 and event["detected"] is False
    jpeg = client.app.state.session.latest_jpeg()
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    client.post("/api/stop")


def test_threshold_roundtrip_and_validation(client):
    assert client.post("/api/threshold", json={"value": 0.7}).status_code == 200
    assert client.post("/api/threshold", json={"value": 7}).status_code == 400
