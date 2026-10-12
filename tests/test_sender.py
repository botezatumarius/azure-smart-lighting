"""Check real HTTP transport and distinguish transport failures from empty scenes."""

import csv
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from smart_lighting import sender


@pytest.fixture
def receiver():
    observations = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            metadata = parse_qs(urlsplit(self.path).query)
            observations.append((metadata, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "frame_id": metadata["frame_id"][0],
                        "person_count": 3,
                        "server_processing_ms": 12,
                        "inference_ms": 10,
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", observations
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def settings(tmp_path, endpoint):
    return sender.SendConfig(
        "https://example.org/camera.jpg", "snapshot", "test", 1, 5, endpoint, tmp_path / "client"
    )


def test_sender_preserves_payload_hash_and_server_result(tmp_path, monkeypatch, receiver):
    endpoint, observations = receiver
    monkeypatch.setattr(sender, "capture_frame", lambda *args: np.zeros((40, 60, 3), np.uint8))
    output = sender.send(settings(tmp_path, endpoint))
    with (output / "client.csv").open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    metadata, body = observations[0]
    assert row["status"] == "ok"
    assert row["person_count"] == "3"
    assert row["payload_sha256"] == hashlib.sha256(body).hexdigest()
    assert "video_time_s" not in metadata
    assert float(row["end_to_end_ms"]) >= float(row["response_ms"])
    assert (output / "raw" / f"{row['frame_id']}.png").is_file()


def test_network_failure_is_recorded_without_a_person_count(tmp_path, monkeypatch):
    monkeypatch.setattr(sender, "capture_frame", lambda *args: np.zeros((40, 60, 3), np.uint8))

    def fail(*args, **kwargs):
        raise TimeoutError("test timeout")

    monkeypatch.setattr(sender, "urlopen", fail)
    config = settings(tmp_path, "http://127.0.0.1:8000")
    with pytest.raises(TimeoutError):
        sender.send(config)
    with (config.output / "client.csv").open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    assert row["status"] == "error"
    assert row["person_count"] == ""
    assert json.loads((config.output / "run.json").read_text())["status"] == "failed"
