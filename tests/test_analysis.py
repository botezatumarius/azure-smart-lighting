"""Protect evidence joins and independently labelled count metrics."""

import csv
import json

import pytest

from scripts.analyze_vm import analyze


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def experiment(tmp_path):
    vm_dir = tmp_path / "vm"
    client_dir = tmp_path / "client"
    vm_dir.mkdir()
    client_dir.mkdir()
    server_rows = [
        {
            "frame_id": "frame-1",
            "received_utc": "2026-10-13T10:00:00+00:00",
            "captured_utc": "2026-10-13T10:00:00+00:00",
            "video_time_s": "",
            "person_count": 2,
            "payload_sha256": "hash-1",
            "server_processing_ms": 10,
            "inference_ms": 8,
        },
        {
            "frame_id": "frame-2",
            "received_utc": "2026-10-13T10:00:05+00:00",
            "captured_utc": "2026-10-13T10:00:05+00:00",
            "video_time_s": "",
            "person_count": 4,
            "payload_sha256": "hash-2",
            "server_processing_ms": 20,
            "inference_ms": 16,
        },
    ]
    write_csv(vm_dir / "counts.csv", server_rows)
    (vm_dir / "resources.csv").write_text("timestamp_utc,sample_seconds\n", encoding="utf-8")
    write_csv(
        client_dir / "client.csv",
        [
            {
                "frame_id": row["frame_id"],
                "status": "ok",
                "payload_sha256": row["payload_sha256"],
                "payload_bytes": 100,
                "response_ms": 30,
                "end_to_end_ms": 40,
            }
            for row in server_rows
        ],
    )
    (client_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "input_type": "live_webcam",
                "config": {"jpeg_quality": 95, "interval": 5},
            }
        ),
        encoding="utf-8",
    )
    return vm_dir, client_dir


def test_manual_labels_measure_count_agreement_and_are_preserved(experiment):
    vm_dir, client_dir = experiment
    report = analyze(vm_dir, client_dir)
    assert report["successful_frames"] == 2
    labels = client_dir / "analysis/labels.csv"
    write_csv(
        labels,
        [{"frame_id": "frame-1", "human_count": 2}, {"frame_id": "frame-2", "human_count": 6}],
    )
    report = analyze(vm_dir, client_dir, labels)
    assert report["count_effectiveness"]["mean_absolute_count_error"] == 1
    assert report["count_effectiveness"]["exact_count_match_percent"] == 50
    assert read_labels(labels)[1]["human_count"] == "6"


def read_labels(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_mismatched_client_and_vm_payloads_cannot_be_summarised(experiment):
    vm_dir, client_dir = experiment
    csv_path = vm_dir / "counts.csv"
    csv_path.write_text(csv_path.read_text().replace("hash-2", "wrong-hash"), encoding="utf-8")
    with pytest.raises(ValueError, match="checksums differ"):
        analyze(vm_dir, client_dir)
