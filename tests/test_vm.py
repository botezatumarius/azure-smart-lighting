import csv

import numpy as np
import pytest
from fastapi.testclient import TestClient

from smart_lighting.camera import to_jpeg
from smart_lighting.detector import Detection
from smart_lighting.vm import create_app

TOKEN = "secret"
JPEG = to_jpeg(np.zeros((48, 64, 3), dtype=np.uint8))


class FakeDetector:
    def detect(self, frame):
        return Detection(people=4, annotated=frame, inference_ms=50.0)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, TOKEN, FakeDetector())) as client:
        yield client


def upload(client, camera="cam-1", token=TOKEN, body=JPEG):
    return client.post(
        "/frames",
        params={"camera": camera},
        content=body,
        headers={"X-API-Key": token, "Content-Type": "image/jpeg"},
    )


def test_frame_is_counted_and_published(client, tmp_path):
    response = upload(client)

    assert response.json() == {"people": 4, "inference_ms": 50.0}
    with (tmp_path / "counts.csv").open() as file:
        assert [row["people"] for row in csv.DictReader(file)] == ["4"]
    assert client.get("/latest.jpg").content.startswith(b"\xff\xd8")  # JPEG magic number
    assert "people" in client.get("/counts.csv").text


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"token": "wrong"}, 401),
        ({"body": b"not an image"}, 400),
        ({"camera": "../escape"}, 422),
    ],
)
def test_bad_uploads_are_rejected(client, kwargs, status):
    assert upload(client, **kwargs).status_code == status


def test_dashboard_is_served(client):
    assert "Smart Lighting" in client.get("/").text
    assert client.get("/dashboard.js").status_code == 200
