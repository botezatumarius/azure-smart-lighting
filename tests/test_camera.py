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
