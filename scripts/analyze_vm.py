"""Summarise a paired sender/VM experiment; never infer accuracy from model confidence."""

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

from smart_lighting.plotting import plot_counts


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def statistics(values) -> dict:
    values = sorted(float(value) for value in values)
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean": round(mean(values), 3),
        "p95_nearest_rank": round(values[math.ceil(0.95 * len(values)) - 1], 3),
        "max": round(max(values), 3),
    }


def analyze(vm_dir: Path, client_dir: Path, labels: Path | None = None) -> dict:
    client = read_csv(client_dir / "client.csv")
    successful = [row for row in client if row["status"] == "ok"]
    if not successful:
        raise ValueError("No successful client observations.")
    ids = {row["frame_id"] for row in successful}
    counts = [row for row in read_csv(vm_dir / "counts.csv") if row["frame_id"] in ids]
    if len(counts) != len(ids) or len(ids) != len(successful):
        raise ValueError("Successful client observations must match exactly one VM row each.")
    by_id = {row["frame_id"]: row for row in counts}
    if any(row["payload_sha256"] != by_id[row["frame_id"]]["payload_sha256"] for row in successful):
        raise ValueError("Client and VM payload checksums differ.")
    manifest = json.loads((client_dir / "run.json").read_text(encoding="utf-8"))
    first = min(datetime.fromisoformat(row["received_utc"]) for row in counts)
    last = max(
        datetime.fromisoformat(row["received_utc"])
        + timedelta(milliseconds=float(row["server_processing_ms"]))
        for row in counts
    )
    resources = []
    for row in read_csv(vm_dir / "resources.csv"):
        end = datetime.fromisoformat(row["timestamp_utc"])
        start = end - timedelta(seconds=float(row["sample_seconds"]))
        if start <= last and end >= first:
            resources.append(row)
    payload_bytes = sum(int(row["payload_bytes"]) for row in successful)
    report = {
        "note": "Record deployment location, Azure configuration and cost separately.",
        "client_status": manifest["status"],
        "successful_frames": len(successful),
        "failed_attempts": sum(row["status"] != "ok" for row in client),
        "input_type": manifest["input_type"],
        "jpeg_quality": manifest["config"]["jpeg_quality"],
        "server_window_start_utc": first.isoformat(),
        "server_window_end_utc": last.isoformat(),
        "payload_bytes_total": payload_bytes,
        "payload_bytes_per_frame_mean": round(payload_bytes / len(successful), 3),
        "nominal_payload_kbit_per_second": round(
            payload_bytes / len(successful) * 8 / manifest["config"]["interval"] / 1000, 3
        ),
        "response_ms": statistics(row["response_ms"] for row in successful),
        "end_to_end_ms": statistics(row["end_to_end_ms"] for row in successful),
        "inference_ms": statistics(row["inference_ms"] for row in counts),
        "server_processing_ms": statistics(row["server_processing_ms"] for row in counts),
        "resource_window_note": "Full 1-second samples overlapping the server processing window; "
        "boundary samples include nearby idle time. Host network counters "
        "include all interfaces, SSH, dashboard and unrelated traffic.",
        "resource_samples": len(resources),
        "host_cpu_percent": statistics(row["host_cpu_percent"] for row in resources),
        "process_cpu_core_percent": statistics(
            row["process_cpu_core_percent"] for row in resources
        ),
        "process_rss_bytes": statistics(row["process_rss_bytes"] for row in resources),
        "host_memory_percent": statistics(row["host_memory_percent"] for row in resources),
        "cost": "Pending Azure Cost Management export; student credit consumption is still cost.",
    }
    summary_dir = client_dir / "analysis"
    summary_dir.mkdir(exist_ok=True)
    paired = summary_dir / "vm-counts.csv"
    with paired.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    plot_counts(paired, summary_dir / "people-over-time.png")
    template = summary_dir / "labels.csv"
    if not template.exists():
        with template.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("frame_id", "human_count"))
            writer.writerows((row["frame_id"], "") for row in counts)
    if labels:
        annotations = [row for row in read_csv(labels) if row["human_count"] != ""]
        labelled_ids = [row["frame_id"] for row in annotations]
        if len(labelled_ids) != len(set(labelled_ids)) or not set(labelled_ids) <= ids:
            raise ValueError("Labels must have unique frame IDs from this experiment.")
        human_counts = [int(row["human_count"]) for row in annotations]
        if not human_counts or min(human_counts) < 0:
            raise ValueError("Provide at least one non-negative integer human count.")
        errors = [
            abs(int(by_id[row["frame_id"]]["person_count"]) - human)
            for row, human in zip(annotations, human_counts, strict=True)
        ]
        report["count_effectiveness"] = {
            "labelled_frames": len(errors),
            "total_frames": len(counts),
            "mean_absolute_count_error": round(mean(errors), 3),
            "exact_count_match_percent": round(100 * errors.count(0) / len(errors), 3),
            "note": "Count agreement only; does not establish bounding-box precision or recall.",
        }
    else:
        report["count_effectiveness"] = "Pending independent manual labels in analysis/labels.csv."
    (summary_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vm_dir", type=Path)
    parser.add_argument("client_dir", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.vm_dir, args.client_dir, args.labels), indent=2))
