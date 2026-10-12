import cv2
import numpy as np
import pytest

from smart_lighting import capture


@pytest.fixture
def recorded_video(tmp_path):
    path = tmp_path / "test clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (48, 32))
    assert writer.isOpened()
    try:
        for index in range(24):
            writer.write(np.full((32, 48, 3), index * 8, dtype=np.uint8))
    finally:
        writer.release()
    return str(path)


@pytest.mark.parametrize(("position_s", "expected"), [(0, 0), (0.5, 32), (1.5, 96)])
def test_recorded_video_samples_the_requested_time(recorded_video, position_s, expected):
    frame = capture.capture_video_frame(recorded_video, position_s)
    assert abs(float(frame.mean()) - expected) < 3


def test_recorded_video_stops_at_its_end(recorded_video):
    with pytest.raises(EOFError):
        capture.capture_video_frame(recorded_video, 3)


def test_unreadable_video_is_an_error(tmp_path):
    path = tmp_path / "invalid.avi"
    path.write_bytes(b"not a video")
    with pytest.raises(OSError, match="Cannot open"):
        capture.capture_video_frame(str(path), 0)


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
