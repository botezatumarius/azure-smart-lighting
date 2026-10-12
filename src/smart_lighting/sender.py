"""Capture locally and upload JPEG frames through a local SSH tunnel. No local YOLO."""

import csv
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import cv2

from smart_lighting.capture import capture_frame, capture_video_frame
from smart_lighting.scheduling import next_slot

FIELDS = (
    "frame_id",
    "camera_id",
    "captured_utc",
    "input_type",
    "video_time_s",
    "status",
    "error",
    "person_count",
    "capture_ms",
    "encode_ms",
    "response_ms",
    "end_to_end_ms",
    "schedule_lag_ms",
    "payload_bytes",
    "payload_sha256",
    "server_processing_ms",
    "inference_ms",
)


@dataclass(frozen=True)
class SendConfig:
    source: str
    source_kind: str
    camera_id: str
    frames: int
    interval: float
    endpoint: str
    output: Path
    jpeg_quality: int = 95
    timeout: float = 30


def validate_endpoint(endpoint: str) -> None:
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("Use http://127.0.0.1:8000 via an SSH tunnel (or a local test server).")


def send(config: SendConfig) -> Path:
    validate_endpoint(config.endpoint)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", config.camera_id):
        raise ValueError("Camera ID must contain 1-64 letters, digits, underscores or hyphens.")
    config.output.mkdir(parents=True, exist_ok=False)
    (config.output / "raw").mkdir()
    is_video = config.source_kind == "video"
    run_id = uuid.uuid4().hex
    manifest = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "run_id": run_id,
        "status": "running",
        "completed_frames": 0,
        "started_utc": datetime.now(UTC).isoformat(),
        "input_type": "recorded_video_test" if is_video else "live_webcam",
        "sampling_clock": "monotonic real time, including recorded-video replay",
        "response_ms_definition": "HTTP upload start to complete JSON response, client clock",
        "end_to_end_ms_definition": "capture start to response receipt, includes JPEG encoding",
        "retry_policy": "No automatic POST retries; inspect VM CSV after an ambiguous timeout.",
    }
    if is_video:
        with Path(config.source).open("rb") as stream:
            manifest["source_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    slot = time.monotonic()
    try:
        with (config.output / "client.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            manifest["completion_reason"] = "frame_limit"
            for index in range(config.frames):
                time.sleep(max(0, slot - time.monotonic()))
                started = time.monotonic()
                frame_id = f"{run_id}-{index + 1:06d}"
                row = {
                    "frame_id": frame_id,
                    "camera_id": config.camera_id,
                    "input_type": manifest["input_type"],
                    "video_time_s": index * config.interval if is_video else None,
                    "schedule_lag_ms": round((started - slot) * 1000, 3),
                }
                try:
                    if is_video:
                        try:
                            frame = capture_video_frame(config.source, row["video_time_s"])
                        except EOFError:
                            manifest["completion_reason"] = "end_of_video"
                            break
                    else:
                        frame = capture_frame(config.source, config.source_kind)
                    acquired = time.monotonic()
                    row["captured_utc"] = datetime.now(UTC).isoformat(timespec="milliseconds")
                    success, image = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
                    )
                    if not success:
                        raise OSError("Could not encode captured frame.")
                    payload = image.tobytes()
                    encoded = time.monotonic()
                    metadata = {
                        key: row[key]
                        for key in (
                            "frame_id",
                            "camera_id",
                            "captured_utc",
                            "input_type",
                            "video_time_s",
                        )
                        if row[key] is not None
                    }
                    request = Request(
                        config.endpoint.rstrip("/") + "/frames?" + urlencode(metadata),
                        data=payload,
                        headers={"Content-Type": "image/jpeg"},
                        method="POST",
                    )
                    upload_started = time.monotonic()
                    with urlopen(request, timeout=config.timeout) as response:
                        result = json.load(response)
                    finished = time.monotonic()
                    if result["frame_id"] != frame_id:
                        raise ValueError("VM returned a result for a different frame.")
                    if not cv2.imwrite(str(config.output / "raw" / f"{frame_id}.png"), frame):
                        raise OSError("Could not preserve the local raw frame.")
                    row.update(
                        status="ok",
                        person_count=result["person_count"],
                        capture_ms=round((acquired - started) * 1000, 3),
                        encode_ms=round((encoded - acquired) * 1000, 3),
                        response_ms=round((finished - upload_started) * 1000, 3),
                        end_to_end_ms=round((finished - started) * 1000, 3),
                        payload_bytes=len(payload),
                        payload_sha256=hashlib.sha256(payload).hexdigest(),
                        server_processing_ms=result["server_processing_ms"],
                        inference_ms=result["inference_ms"],
                    )
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    writer.writerow(row)
                    stream.flush()
                    raise
                writer.writerow(row)
                stream.flush()
                manifest["completed_frames"] += 1
                logging.info(
                    "%s | people=%s | response=%.1f ms",
                    frame_id,
                    row["person_count"],
                    row["response_ms"],
                )
                following = next_slot(slot, time.monotonic(), config.interval)
                if following - slot > config.interval * 1.01:
                    logging.warning(
                        "Missed capture slots skipped: processing exceeded the interval."
                    )
                slot = following
        if not manifest["completed_frames"]:
            raise ValueError("The source provided no observations.")
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["ended_utc"] = datetime.now(UTC).isoformat()
        (config.output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return config.output
