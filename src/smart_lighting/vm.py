"""Single-model VM service. Bind to loopback; publish only read routes through Nginx."""

import csv
import hashlib
import io
import json
import logging
import math
import platform
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

from smart_lighting.detection import PersonDetector
from smart_lighting.metrics import ResourceMonitor

MAX_UPLOAD_BYTES = 2_000_000
MAX_PIXELS = 4096 * 4096
WEB = Path(__file__).with_name("web")
FIELDS = (
    "received_utc",
    "captured_utc",
    "camera_id",
    "frame_id",
    "input_type",
    "video_time_s",
    "person_count",
    "width",
    "height",
    "payload_bytes",
    "payload_sha256",
    "inference_ms",
    "server_processing_ms",
)
Identifier = Annotated[str, Query(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


@dataclass(frozen=True)
class VMConfig:
    output: Path
    model: Path
    confidence: float = 0.5
    image_size: int = 640
    threads: int = 1


class FrameProcessor:
    def __init__(self, config: VMConfig):
        self.config = config
        config.output.mkdir(parents=True, exist_ok=False)
        for folder in ("raw", "annotated", "detections"):
            (config.output / folder).mkdir()
        self.lock = threading.Lock()
        self.rows = deque(maxlen=300)
        self.seen: set[str] = set()
        self.detector = PersonDetector(config.model, config.confidence, config.image_size)
        # Ultralytics sets its own thread count on first use.
        self.detector.warm_up()
        torch.set_num_threads(config.threads)
        self.detector.warm_up()
        self.monitor = ResourceMonitor(config.output / "resources.csv")
        self.manifest = {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "started_utc": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": psutil.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "memory_total_bytes": psutil.virtual_memory().total,
            "versions": {
                name: version(name)
                for name in ("torch", "ultralytics", "opencv-python", "fastapi", "psutil")
            },
            "model_sha256": hashlib.sha256(config.model.read_bytes()).hexdigest(),
            "status": "running",
            "completed_frames": 0,
            "server_processing_ms_definition": (
                "body receipt + decode + inference + images/detection JSON; "
                "excludes CSV flush, dashboard update and response transport"
            ),
            "resource_sampling": "1 second after warm-up; whole process and whole host",
        }
        with (config.output / "counts.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
        self._manifest()
        self.monitor.start()

    def _manifest(self) -> None:
        (self.config.output / "run.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        self.monitor.close()
        self.manifest.update(status="stopped", ended_utc=datetime.now(UTC).isoformat())
        self._manifest()

    def process(self, payload: bytes, metadata: dict, started: float) -> dict:
        # Single model and writer; reject when busy.
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, "Detector busy; this frame was not processed.")
        try:
            frame_id = metadata["frame_id"]
            if frame_id in self.seen:
                raise HTTPException(409, "Frame ID already processed.")
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    if (
                        image.format != "JPEG"
                        or max(image.size) > 4096
                        or image.width * image.height > MAX_PIXELS
                    ):
                        raise ValueError("Expected a JPEG no larger than 4096 x 4096 pixels.")
                    image.verify()
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as error:
                raise HTTPException(400, "Invalid or oversized JPEG.") from error
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(400, "JPEG could not be decoded.")
            torch.set_num_threads(self.config.threads)
            inference_start = time.monotonic()
            result = self.detector.detect(frame)
            inference_ms = (time.monotonic() - inference_start) * 1000
            success, annotated = cv2.imencode(".jpg", result.plot())
            if not success:
                raise OSError("Could not encode annotated image.")
            output = self.config.output
            (output / "raw" / f"{frame_id}.jpg").write_bytes(payload)
            (output / "annotated" / f"{frame_id}.jpg").write_bytes(annotated.tobytes())
            (output / "detections" / f"{frame_id}.json").write_text(
                json.dumps(
                    {
                        **metadata,
                        "boxes_xyxy": result.boxes.xyxy.cpu().tolist(),
                        "confidences": result.boxes.conf.cpu().tolist(),
                    }
                ),
                encoding="utf-8",
            )
            row = {
                **metadata,
                "person_count": len(result.boxes),
                "width": frame.shape[1],
                "height": frame.shape[0],
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "inference_ms": round(inference_ms, 3),
                "server_processing_ms": round((time.monotonic() - started) * 1000, 3),
            }
            with (output / "counts.csv").open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
            self.rows.append(row)
            self.seen.add(frame_id)
            self.manifest["completed_frames"] += 1
            self._manifest()
            logging.info(
                "%s | people=%s | inference=%.1f ms", frame_id, row["person_count"], inference_ms
            )
            return row
        finally:
            self.lock.release()


def create_app(config: VMConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.processor = FrameProcessor(config)
        try:
            yield
        finally:
            app.state.processor.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
                "style-src 'unsafe-inline'; img-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'",
            }
        )
        return response

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/dashboard.js")
    def javascript():
        return FileResponse(WEB / "dashboard.js", media_type="text/javascript")

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.get("/results")
    def results():
        processor = app.state.processor
        with processor.lock:
            return {
                "rows": list(processor.rows),
                "total_frames": len(processor.seen),
                "resources": processor.monitor.latest,
            }

    @app.get("/counts.csv")
    def counts():
        processor = app.state.processor
        with processor.lock:
            return Response((config.output / "counts.csv").read_bytes(), media_type="text/csv")

    @app.get("/latest.jpg")
    def latest():
        processor = app.state.processor
        with processor.lock:
            if not processor.rows:
                raise HTTPException(404, "No frames received yet.")
            path = config.output / "annotated" / f"{processor.rows[-1]['frame_id']}.jpg"
        return FileResponse(path)

    @app.post("/frames")
    async def upload(
        request: Request,
        frame_id: Identifier,
        camera_id: Identifier,
        captured_utc: datetime,
        input_type: Annotated[str, Query(pattern=r"^(live_webcam|recorded_video_test)$")],
        video_time_s: Annotated[float | None, Query(ge=0)] = None,
    ):
        started = time.monotonic()
        received = datetime.now(UTC).isoformat(timespec="milliseconds")
        if captured_utc.utcoffset() is None:
            raise HTTPException(422, "captured_utc must include a timezone.")
        if (input_type == "recorded_video_test") != (video_time_s is not None):
            raise HTTPException(422, "Video inputs require video_time_s; live inputs must omit it.")
        if video_time_s is not None and not math.isfinite(video_time_s):
            raise HTTPException(422, "video_time_s must be finite.")
        if request.headers.get("content-type", "").split(";")[0] != "image/jpeg":
            raise HTTPException(415, "Expected image/jpeg.")
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "JPEG exceeds the 2 MB upload limit.")
            payload.extend(chunk)
        metadata = {
            "received_utc": received,
            "captured_utc": captured_utc.astimezone(UTC).isoformat(),
            "camera_id": camera_id,
            "frame_id": frame_id,
            "input_type": input_type,
            "video_time_s": video_time_s,
        }
        return await run_in_threadpool(
            app.state.processor.process, bytes(payload), metadata, started
        )

    return app
