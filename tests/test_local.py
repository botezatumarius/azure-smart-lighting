"""Protect the measurement and evidence behaviour without calling a real webcam."""

import argparse
import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from smart_lighting import local
from smart_lighting.cli import parser, positive_float


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_interval_rejects_nonpositive_and_nonfinite_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(value)


def test_schedule_does_not_add_inference_time_to_each_interval():
    assert local.next_slot(100, 101.2, 5) == 105
    assert local.next_slot(100, 112, 5) == 115


def test_assignment_defaults():
    args = parser().parse_args(["local", "--source", "https://example.org/camera.mjpg"])
    assert args.interval == 5
    assert args.confidence == 0.5
    assert args.model.name == "yolov8n.pt"


class FakeDetector:
    def __init__(self, *args):
        pass

    def warm_up(self):
        pass

    def detect(self, frame):
        # Zero detections still count as an observation.
        return SimpleNamespace(
            boxes=FakeBoxes(),
            plot=lambda: frame,
        )


class FakeBoxes:
    xyxy = SimpleNamespace(cpu=lambda: np.empty((0, 4)))
    conf = SimpleNamespace(cpu=lambda: np.empty(0))

    def __len__(self):
        return 0


def config(tmp_path, frames=1):
    model = tmp_path / "yolov8n.pt"
    model.write_bytes(b"test model, not actual weights")
    return local.RunConfig(
        "https://example.org/camera.jpg",
        "snapshot",
        "test",
        frames,
        5.0,
        0.5,
        640,
        model,
        tmp_path / "run",
    )


def test_run_preserves_zero_counts_and_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "PersonDetector", FakeDetector)
    monkeypatch.setattr(local, "capture_frame", lambda *args: np.zeros((40, 60, 3), np.uint8))
    output = local.run(config(tmp_path))
    with (output / "counts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["person_count"] == "0"
    assert rows[0]["timestamp_utc"].endswith("+00:00")
    assert rows[0]["width"] == "60"
    assert (output / "raw/000001.png").is_file()
    assert (output / "annotated/000001.jpg").is_file()
    assert (output / "people-over-time.png").is_file()
    assert json.loads((output / "run.json").read_text())["status"] == "completed"


def test_capture_failure_is_not_recorded_as_zero_people(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "PersonDetector", FakeDetector)

    def unavailable(*args):
        raise OSError("Camera unavailable")

    monkeypatch.setattr(local, "capture_frame", unavailable)
    settings = config(tmp_path)
    with pytest.raises(OSError, match="Camera unavailable"):
        local.run(settings)
    manifest = json.loads((settings.output / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_frames"] == 0
    with (settings.output / "counts.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []


def test_run_never_overwrites_an_existing_experiment(tmp_path):
    settings = config(tmp_path)
    settings.output.mkdir()
    with pytest.raises(FileExistsError):
        local.run(settings)


def test_recorded_video_uses_media_time_and_stops_normally(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "PersonDetector", FakeDetector)
    source = tmp_path / "test.avi"
    source.write_bytes(b"test source for hashing")
    positions = []

    def read(source, position_s):
        positions.append(position_s)
        if position_s >= 10:
            raise EOFError
        return np.full((40, 60, 3), position_s, dtype=np.uint8)

    def unexpected_sleep(seconds):
        raise AssertionError("Offline video processing must not wait for live capture slots.")

    monkeypatch.setattr(local, "capture_video_frame", read)
    monkeypatch.setattr(local.time, "sleep", unexpected_sleep)
    settings = replace(config(tmp_path, frames=60), source=str(source), source_kind="video")
    output = local.run(settings)
    with (output / "counts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert positions == [0, 5, 10]
    assert [float(row["video_time_s"]) for row in rows] == [0, 5]
    assert all(row["schedule_lag_ms"] == "" for row in rows)
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["input_type"] == "recorded_video_test"
    assert metadata["status"] == "completed"
    assert metadata["completion_reason"] == "end_of_video"
    assert metadata["completed_frames"] == 2
    assert (output / "people-over-time.png").is_file()
