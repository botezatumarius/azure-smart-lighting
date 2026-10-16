"""YOLOv8 person detection, shared by the local and cloud pipelines."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PERSON = 0  # COCO class id


@dataclass(frozen=True)
class Detection:
    people: int
    annotated: np.ndarray
    inference_ms: float


class PersonDetector:
    def __init__(
        self,
        model: Path = Path("models/yolov8n.pt"),
        confidence: float = 0.5,
        # Native camera width; at 640 px distant people are missed.
        image_size: int = 1280,
    ):
        self.model = YOLO(model)
        self.confidence = confidence
        self.image_size = image_size
        # Warm-up run.
        self.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> Detection:
        result = self.model.predict(
            frame, classes=[PERSON], conf=self.confidence, imgsz=self.image_size, verbose=False
        )[0]
        return Detection(len(result.boxes), result.plot(), sum(result.speed.values()))
