import csv
from datetime import datetime

import numpy as np

from smart_lighting.camera import capture
from smart_lighting.detector import Detection
from smart_lighting.local import run

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeDetector:
    def detect(self, frame):
        return Detection(people=2, annotated=frame, inference_ms=12.3)


def test_run_saves_counts_images_and_graph(tmp_path):
    frames = [(datetime(2026, 1, 1, 12, 0, s).astimezone(), FRAME) for s in (0, 5, 10)]

    run(frames, FakeDetector(), "test-cam", tmp_path)

    with (tmp_path / "counts.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [(row["camera"], row["people"]) for row in rows] == [("test-cam", "2")] * 3
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3
    assert (tmp_path / "people-over-time.png").exists()


def test_capture_skips_frames_the_camera_fails_to_deliver():
    results = iter([ConnectionError("offline"), FRAME])

    def read(url):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    frames = list(capture("http://camera", interval=0, frames=2, read=read))

    assert len(frames) == 1
    assert frames[0][1] is FRAME
