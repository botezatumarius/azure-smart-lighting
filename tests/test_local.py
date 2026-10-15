import csv

import numpy as np

from smart_lighting.detector import Detection
from smart_lighting.local import run

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeDetector:
    def detect(self, frame):
        return Detection(people=2, annotated=frame, inference_ms=12.3)


def test_run_saves_counts_images_and_graph(tmp_path):
    run(lambda: FRAME, FakeDetector(), "test-cam", tmp_path, interval=0, frames=3)

    with (tmp_path / "counts.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [(row["camera"], row["people"]) for row in rows] == [("test-cam", "2")] * 3
    assert list((tmp_path / "frames").glob("*.jpg"))
    assert (tmp_path / "people-over-time.png").exists()


def test_run_skips_frames_the_camera_fails_to_deliver(tmp_path):
    results = iter([ConnectionError("offline"), FRAME])

    def read():
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    run(read, FakeDetector(), "test-cam", tmp_path, interval=0, frames=2)

    with (tmp_path / "counts.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 1
