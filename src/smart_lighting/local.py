"""Part 1 orchestration, based on the teaching team's grabber.py pipeline."""

import csv
import hashlib
import json
import logging
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import cv2

from smart_lighting.capture import capture_frame
from smart_lighting.detection import PersonDetector
from smart_lighting.plotting import plot_counts

LOGGER = logging.getLogger(__name__)
FIELDS = (
    "timestamp_utc",
    "camera_id",
    "frame_id",
    "person_count",
    "width",
    "height",
    "capture_ms",
    "inference_ms",
    "processing_ms",
    "schedule_lag_ms",
    "frame_sha256",
)


@dataclass(frozen=True)
class RunConfig:
    source: str
    source_kind: str
    camera_id: str
    frames: int
    interval: float
    confidence: float
    image_size: int
    model: Path
    output: Path


def next_slot(current_slot: float, now: float, interval: float) -> float:
    """Keep the original cadence; skip missed slots instead of creating a burst."""
    return current_slot + max(1, math.ceil((now - current_slot) / interval)) * interval


def save_image(path: Path, frame) -> None:
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write image: {path}")


def run(config: RunConfig) -> Path:
    # Never overwrite a previous run.
    config.output.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "annotated"):
        (config.output / name).mkdir()
    metadata = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "started_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("ultralytics", "torch", "opencv-python")},
        "status": "initialising",
        "completed_frames": 0,
        "timestamp_definition": (
            "UTC at local frame acquisition completion; not camera exposure time"
        ),
        "processing_ms_definition": (
            "capture + inference + image/JSONL persistence; excludes CSV flush"
        ),
    }
    manifest = config.output / "run.json"
    csv_path = config.output / "counts.csv"
    try:
        detector = PersonDetector(config.model, config.confidence, config.image_size)
        detector.warm_up()
        metadata["model_sha256"] = hashlib.sha256(config.model.read_bytes()).hexdigest()
        metadata["status"] = "running"
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        slot = time.monotonic()
        previous_hash = None
        with (
            csv_path.open("w", newline="", encoding="utf-8") as stream,
            (config.output / "detections.jsonl").open("w", encoding="utf-8") as detections,
        ):
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            stream.flush()
            for index in range(config.frames):
                time.sleep(max(0, slot - time.monotonic()))
                started = time.monotonic()
                frame = capture_frame(config.source, config.source_kind)
                acquired = time.monotonic()
                timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")
                result = detector.detect(frame)
                inferred = time.monotonic()
                frame_id = f"{index + 1:06d}"
                digest = hashlib.sha256(frame.tobytes()).hexdigest()
                if digest == previous_hash:
                    LOGGER.warning(
                        "Frame %s is identical to the previous image; check feed freshness.",
                        frame_id,
                    )
                previous_hash = digest
                save_image(config.output / "raw" / f"{frame_id}.png", frame)
                save_image(config.output / "annotated" / f"{frame_id}.jpg", result.plot())
                boxes = result.boxes
                count = len(boxes)
                detections.write(
                    json.dumps(
                        {
                            "frame_id": frame_id,
                            "timestamp_utc": timestamp,
                            "boxes_xyxy": boxes.xyxy.cpu().tolist(),
                            "confidences": boxes.conf.cpu().tolist(),
                        }
                    )
                    + "\n"
                )
                detections.flush()
                finished = time.monotonic()
                writer.writerow(
                    {
                        "timestamp_utc": timestamp,
                        "camera_id": config.camera_id,
                        "frame_id": frame_id,
                        "person_count": count,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "capture_ms": round((acquired - started) * 1000, 3),
                        "inference_ms": round((inferred - acquired) * 1000, 3),
                        "processing_ms": round((finished - started) * 1000, 3),
                        "schedule_lag_ms": round((started - slot) * 1000, 3),
                        "frame_sha256": digest,
                    }
                )
                stream.flush()
                metadata["completed_frames"] = index + 1
                LOGGER.info(
                    "Frame %s | people=%d | inference=%.0f ms",
                    frame_id,
                    count,
                    (inferred - acquired) * 1000,
                )
                following = next_slot(slot, time.monotonic(), config.interval)
                if following - slot > config.interval * 1.01:
                    LOGGER.warning(
                        "Processing exceeded the interval; missed capture slots were skipped."
                    )
                slot = following
        plot_counts(csv_path, config.output / "people-over-time.png")
        metadata["status"] = "completed"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        raise
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if metadata["completed_frames"] and metadata["status"] != "completed":
            try:
                plot_counts(csv_path, config.output / "people-over-time.png")
            except Exception:
                LOGGER.exception("Could not plot partial results.")
        metadata["ended_utc"] = datetime.now(UTC).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return config.output
