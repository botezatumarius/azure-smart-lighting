import cv2
import numpy as np
import pytest

from smart_lighting import capture


def test_retry_recovers_from_snapshot_being_rewritten(monkeypatch):
    attempts = []
    expected = np.zeros((10, 10, 3), np.uint8)

    def read(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("Incomplete JPEG")
        return expected

    monkeypatch.setattr(capture, "_capture_frame", read)
    monkeypatch.setattr(capture.time, "sleep", lambda seconds: None)
    assert capture.capture_frame("https://example.org/camera.jpg", "snapshot") is expected
    assert len(attempts) == 2


def test_permanent_failure_stops_after_three_attempts(monkeypatch):
    attempts = []

    def read(*args):
        attempts.append(1)
        raise cv2.error("Invalid image")

    monkeypatch.setattr(capture, "_capture_frame", read)
    monkeypatch.setattr(capture.time, "sleep", lambda seconds: None)
    with pytest.raises(cv2.error):
        capture.capture_frame("https://example.org/camera.jpg", "snapshot")
    assert len(attempts) == 3
