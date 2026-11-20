from __future__ import annotations

import asyncio
import json
import queue
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio.sources import list_microphones
from .config import AppConfig, load_config
from .errors import DroneError
from .session import ConflictError, DetectionSession, EventHub
from .vision.sources import list_cameras

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class SourceModel(BaseModel):
    kind: Literal["device", "file"]
    device_id: Optional[int] = None
    file_id: Optional[str] = None


class StartModel(BaseModel):
    mode: Literal["audio", "vision"]
    source: SourceModel
    threshold: Optional[float] = None


class ThresholdModel(BaseModel):
    value: float


def _error(message: str, code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=code)


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    hub = EventHub()
    session = DetectionSession(config, hub)
    upload_dir = Path(tempfile.mkdtemp(prefix="dronedetect_"))
    files: dict[str, Path] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        session.stop()
        shutil.rmtree(upload_dir, ignore_errors=True)

    app = FastAPI(title="Обнаружение дронов", lifespan=lifespan)
    app.state.hub = hub
    app.state.session = session
    app.state.config = config

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, exc: ConflictError):
        return _error(exc.message, 409)

    @app.exception_handler(DroneError)
    async def _drone(_: Request, exc: DroneError):
        return _error(exc.message, 400)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        return _error("Некорректный запрос", 400)

    def _status() -> dict:
        return {
            "audio_available": config.audio_model.is_file(),
            "vision_available": config.vision_model.is_file(),
            **session.status(),
        }

    @app.get("/api/status")
    def status():
        return _status()

    @app.get("/api/devices")
    def devices():
        def safe(fn):
            try:
                return [{"id": d.id, "name": d.name} for d in fn()]
            except Exception:  # noqa: BLE001
                return []

        return {"microphones": safe(list_microphones), "cameras": safe(list_cameras)}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        suffix = "".join(c for c in Path(file.filename or "").suffix if c.isalnum() or c == ".")[:10]
        file_id = uuid.uuid4().hex
        path = upload_dir / f"{file_id}{suffix}"
        size = 0
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    path.unlink(missing_ok=True)
                    return _error(f"Файл слишком большой (максимум {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ)", 413)
                out.write(chunk)
        files[file_id] = path
        return {"file_id": file_id}

    @app.post("/api/start")
    def start(body: StartModel):
        src: dict = {"kind": body.source.kind, "device_id": body.source.device_id}
        if body.source.kind == "file":
            path = files.get(body.source.file_id or "")
            if path is None or not path.is_file():
                raise DroneError("Файл не найден, загрузите его заново")
            src["path"] = path
        session.start(body.mode, src, body.threshold)
        return _status()

    @app.post("/api/stop")
    def stop():
        session.stop()
        return _status()

    @app.post("/api/threshold")
    def threshold(body: ThresholdModel):
        session.set_threshold(body.value)
        return {"value": body.value}

    @app.get("/api/events")
    async def events():
        q = hub.subscribe()

        async def gen():
            idle = 0.0
            try:
                while True:
                    try:
                        ev = q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        idle += 0.05
                        if idle >= 15:
                            idle = 0.0
                            yield ": keepalive\n\n"
                        continue
                    idle = 0.0
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.get("/api/video.mjpg")
    async def video():
        async def gen():
            last = -1
            while session.status()["running"]:
                seq = session.frame_seq()
                jpeg = session.latest_jpeg()
                if jpeg is not None and seq != last:
                    last = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                else:
                    await asyncio.sleep(0.02)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
