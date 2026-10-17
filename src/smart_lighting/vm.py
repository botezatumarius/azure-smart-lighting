"""Part 2: web service on the Azure VM.

Receives frames from the local machine, counts people with YOLOv8, stores the results
in CSV files and serves a public results dashboard.
"""

import argparse
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse

from smart_lighting.detector import PersonDetector
from smart_lighting.monitor import RESOURCE_FIELDS, ResourceMonitor
from smart_lighting.results import COUNT_FIELDS, CsvLog

WEB = Path(__file__).with_name("web")
Camera = Annotated[str, Query(pattern=r"^[\w-]{1,32}$")]  # Also used in file names.
Jpeg = Annotated[bytes, Body(media_type="image/jpeg")]


def create_app(data: Path, token: str, detector: PersonDetector) -> FastAPI:
    (data / "frames").mkdir(parents=True, exist_ok=True)
    counts = CsvLog(data / "counts.csv", COUNT_FIELDS)
    latest_image = data / "latest.jpg"
    monitor = ResourceMonitor(CsvLog(data / "resources.csv", RESOURCE_FIELDS))
    # Single model, one frame at a time.
    model_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        monitor.start()
        yield
        monitor.stop()

    app = FastAPI(title="Smart Lighting", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.post("/frames")
    def receive_frame(camera: Camera, image: Jpeg, x_api_key: Annotated[str, Header()] = ""):
        if not secrets.compare_digest(x_api_key, token):
            raise HTTPException(401, "Invalid API key")
        frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Body is not a valid image")
        with model_lock:
            detection = detector.detect(frame)
            timestamp = datetime.now().astimezone()
            jpeg = cv2.imencode(".jpg", detection.annotated)[1].tobytes()
            (data / "frames" / f"{camera}-{timestamp:%Y%m%d-%H%M%S}.jpg").write_bytes(jpeg)
            # Atomic replace so the dashboard never reads a partial file.
            staging = latest_image.with_suffix(".tmp")
            staging.write_bytes(jpeg)
            staging.replace(latest_image)
            counts.append(
                timestamp=timestamp,
                camera=camera,
                people=detection.people,
                inference_ms=detection.inference_ms,
            )
        return {"people": detection.people, "inference_ms": detection.inference_ms}

    @app.get("/")
    def dashboard():
        return FileResponse(WEB / "index.html")

    @app.get("/dashboard.js")
    def dashboard_script():
        return FileResponse(WEB / "dashboard.js")

    @app.get("/counts.csv")
    def counts_csv():
        return FileResponse(counts.path, headers={"Cache-Control": "no-store"})

    @app.get("/latest.jpg")
    def latest():
        if not latest_image.exists():
            raise HTTPException(404, "No frames received yet")
        return FileResponse(latest_image, headers={"Cache-Control": "no-store"})

    @app.get("/resources")
    def resources():
        return monitor.latest

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Smart Lighting VM service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", type=Path, default=Path("runs/vm"), help="results folder")
    args = parser.parse_args()

    token = os.environ["SMART_LIGHTING_TOKEN"]
    uvicorn.run(create_app(args.data, token, PersonDetector()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
