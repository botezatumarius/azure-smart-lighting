"""Exercise the upload boundary and persisted evidence without a real model or Azure."""

import csv
import hashlib
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_local import FakeDetector

from smart_lighting import vm
from smart_lighting.sender import validate_endpoint


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "PersonDetector", FakeDetector)
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake weights")
    output = tmp_path / "vm"
    with TestClient(vm.create_app(vm.VMConfig(output, model))) as client:
        yield client, output


def upload(client, data=None, **changes):
    metadata = {
        "frame_id": "test-000001",
        "camera_id": "test",
        "captured_utc": "2026-10-13T10:00:00+00:00",
        "input_type": "recorded_video_test",
        "video_time_s": 0,
    }
    metadata.update(changes)
    if data is None:
        _, encoded = cv2.imencode(".jpg", np.zeros((40, 60, 3), np.uint8))
        data = encoded.tobytes()
    return client.post(
        "/frames", params=metadata, content=data, headers={"Content-Type": "image/jpeg"}
    )


def test_zero_count_is_preserved_and_duplicate_rejected(service):
    client, output = service
    result = upload(client)
    assert result.status_code == 200
    assert result.json()["person_count"] == 0
    raw = (output / "raw/test-000001.jpg").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result.json()["payload_sha256"]
    assert client.get("/latest.jpg").status_code == 200
    assert upload(client).status_code == 409
    with (output / "counts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["input_type"] == "recorded_video_test"
    assert json.loads((output / "detections/test-000001.json").read_text())["boxes_xyxy"] == []
    assert client.get("/results").json()["total_frames"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"frame_id": "../../escape"},
        {"camera_id": "=formula"},
        {"captured_utc": "2026-10-13T10:00:00"},
        {"video_time_s": "nan"},
        {"input_type": "live_webcam"},
    ],
)
def test_invalid_metadata_cannot_create_observations(service, changes):
    client, output = service
    assert upload(client, **changes).status_code == 422
    assert client.get("/results").json()["total_frames"] == 0
    assert list((output / "raw").iterdir()) == []


def test_invalid_and_oversized_uploads_do_not_become_zero_counts(service):
    client, _ = service
    assert upload(client, b"not an image").status_code == 400
    assert upload(client, b"x" * (vm.MAX_UPLOAD_BYTES + 1)).status_code == 413
    assert client.get("/results").json()["total_frames"] == 0


def test_busy_detector_rejects_work_instead_of_queueing(service):
    client, _ = service
    processor = client.app.state.processor
    with processor.lock:
        assert upload(client).status_code == 429
    assert upload(client).status_code == 200


def test_public_routes_do_not_expose_private_files(service):
    client, _ = service
    for path in ("/run.json", "/resources.csv", "/raw/test.jpg", "/docs", "/openapi.json"):
        assert client.get(path).status_code == 404
    assert client.get("/").status_code == 200
    assert client.get("/dashboard.js").headers["x-content-type-options"] == "nosniff"
    assert client.get("/latest.jpg").status_code == 404


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com:8000",
        "https://127.0.0.1:8000",
        "http://user:secret@localhost",
        "http://127.0.0.1:8000/frames",
        "http://localhost?redirect=example.com",
    ],
)
def test_sender_requires_a_loopback_endpoint(endpoint):
    with pytest.raises(ValueError, match="SSH tunnel"):
        validate_endpoint(endpoint)
