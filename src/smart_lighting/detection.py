"""The same YOLOv8 detector can be used locally and on the Part 2 VM."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class PersonDetector:
    def __init__(self, model: Path, confidence: float = 0.5, image_size: int = 640):
        self.model = YOLO(str(model))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> Results:
        """Return only people: class 0 in the pretrained model's COCO labels."""
        return self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )[0]

    def warm_up(self) -> None:
        """Exclude model initialisation and first-inference overhead from the run."""
        self.detect(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
