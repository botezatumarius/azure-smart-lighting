from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from smart_lighting import camera

JPEG = camera.to_jpeg(np.zeros((48, 64, 3), dtype=np.uint8))
URL = "https://example.org/webcam/image.jpg"


def serve(monkeypatch, data: bytes) -> None:
    monkeypatch.setattr(camera.requests, "get", lambda url, timeout: SimpleNamespace(content=data))


def test_complete_snapshot_is_decoded(monkeypatch):
    serve(monkeypatch, JPEG)
    assert camera.read_frame(URL).shape == (48, 64, 3)


def test_snapshot_cut_short_while_camera_rewrites_it_is_rejected(monkeypatch):
    serve(monkeypatch, JPEG[: len(JPEG) // 2])
    with pytest.raises(ConnectionError):
        camera.read_frame(URL)


def frames_at(*seconds, change_at=()):
    """Grey frames one per given second; a person-sized patch appears from `change_at`."""
    for s in seconds:
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        if s in change_at:
            frame[300:380, 600:640] = 30
        yield datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=s), frame


def test_change_filter_skips_unchanged_frames_and_keeps_changes():
    changes = camera.ChangeFilter()
    passed = [t.second for t, _ in changes(frames_at(0, 1, 2, 3, change_at={2}))]
    assert passed == [0, 2, 3]  # 1 repeats 0; 2 adds a person; 3 shows they left
    assert changes.skipped == 1


def test_change_filter_heartbeat_passes_a_frame_every_minute():
    changes = camera.ChangeFilter(heartbeat=60)
    assert [t.second for t, _ in changes(frames_at(0, 30, 59))] == [0]
    assert len(list(camera.ChangeFilter(heartbeat=60)(frames_at(0, 30, 60)))) == 2
