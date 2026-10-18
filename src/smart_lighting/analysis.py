"""Summarise performance logs into report tables (Markdown) and graphs (Tasks 2.5, 3.6, 4.x).

    python -m smart_lighting.analysis run SENT.csv [--resources R.csv] [--latency COUNTS.csv]
    python -m smart_lighting.analysis load SENT.csv [--resources R.csv] [--latency COUNTS.csv]

    python -m smart_lighting.analysis versus VM_SENT.csv --other SENT.csv --other-latency C.csv

`run` summarises one test run; `load` groups a Part 4 load test by number of cameras
(the camera names n08-c03 encode the level). `--resources` is the VM's resources.csv and
`--latency` the serverless function's counts.csv (storage-to-result latency). `versus` plots
a VM load test against a serverless one (Part 4 comparison).
"""

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["time"] = datetime.fromisoformat(row["timestamp"])
    return rows


def column(rows: list[dict], name: str) -> list[float]:
    return [float(row[name]) for row in rows if row.get(name) not in (None, "")]


def p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0]


def describe(values: list[float]) -> str:
    if not values:
        return "–"
    return f"{statistics.mean(values):.0f} / {statistics.median(values):.0f} / {p95(values):.0f}"


def between(rows: list[dict], start: datetime, end: datetime) -> list[dict]:
    return [row for row in rows if start <= row["time"] <= end]


def summarise(sent: list[dict], resources: list[dict], latency: list[dict], interval: float):
    """One row of metrics for a group of sent frames (one run, or one load level)."""
    start, end = min(r["time"] for r in sent), max(r["time"] for r in sent)
    cameras = {r["camera"] for r in sent}
    # Cameras are staggered by interval / N.
    seconds = (end - start).total_seconds() + interval / len(cameras)
    window = between(resources, start, end + timedelta(seconds=interval))
    expected = len(cameras) * round(seconds / interval)
    return {
        "cameras": len(cameras),
        "frames": len(sent),
        "delivered %": 100 * len(sent) / expected,
        # Serverless: processed = result stored.
        "processed %": 100 * (len(latency) if latency else len(sent)) / expected,
        "frames/s": len(sent) / seconds,
        "kB/frame": statistics.mean(column(sent, "bytes_sent")) / 1000,
        "upload kbit/s": sum(column(sent, "bytes_sent")) * 8 / 1000 / seconds,
        "response ms": column(sent, "response_ms"),
        "inference ms": column(sent, "inference_ms"),
        "latency ms": column(latency, "latency_ms"),
        "people": column(sent, "people") or column(latency, "people"),
        "CPU %": column(window, "cpu_percent"),
        "memory %": column(window, "memory_percent"),
        "service MB": column(window, "service_memory_mb"),
        "net in kbit/s": column(window, "network_in_kbps"),
    }


def markdown(rows: list[dict]) -> str:
    """Numbers as values; lists as mean / median / p95; columns without data are dropped."""

    def cell(value) -> str:
        if isinstance(value, list):
            return describe(value)
        return f"{value:.1f}" if isinstance(value, float) else str(value)

    keys = [k for k in rows[0] if any(not isinstance(r[k], list) or r[k] for r in rows)]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(cell(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines) + "\n\n(lists: mean / median / p95)"


def plot_levels(rows: list[dict], output: Path) -> None:
    metrics = [m for m in ("response ms", "latency ms", "CPU %", "frames/s") if _has(rows, m)]
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(4 * len(metrics), 3.5), layout="constrained"
    )
    cameras = [r["cameras"] for r in rows]
    for ax, metric in zip(axes if len(metrics) > 1 else [axes], metrics, strict=True):
        if isinstance(rows[0][metric], list):
            ax.plot(
                cameras, [statistics.median(r[metric]) for r in rows], marker="o", label="median"
            )
            ax.plot(cameras, [p95(r[metric]) for r in rows], marker="s", label="p95")
            ax.legend()
        else:
            ax.plot(cameras, [r[metric] for r in rows], marker="o", label="achieved")
            ax.plot(cameras, [c / 5 for c in cameras], "--", color="gray", label="offered")
            ax.legend()
        ax.set(title=metric, xlabel="cameras", xscale="log", xticks=cameras, xticklabels=cameras)
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _has(rows: list[dict], metric: str) -> bool:
    return all(not isinstance(r[metric], list) or r[metric] for r in rows)


def plot_versus(vm: list[dict], serverless: list[dict], output: Path) -> None:
    """Frames processed and median end-to-end time vs cameras, for both architectures."""
    figure, (done, time_ax) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    series = {
        "VM (YOLOv8)": (vm, [statistics.median(r["response ms"]) for r in vm]),
        "Serverless (Azure AI)": (
            serverless,
            [
                statistics.median(r["response ms"]) + statistics.median(r["latency ms"])
                for r in serverless
            ],
        ),
    }
    for label, (rows, end_to_end) in series.items():
        cameras = [r["cameras"] for r in rows]
        done.plot(cameras, [r["processed %"] for r in rows], marker="o", label=label)
        time_ax.plot(cameras, [ms / 1000 for ms in end_to_end], marker="o", label=label)
    done.set(title="Frames processed (%)", ylim=(0, 105))
    time_ax.set(title="End-to-end time, median (s)", yscale="log")
    time_ax.axhline(5, color="gray", linestyle="--", label="5 s frame interval")
    ticks = sorted({r["cameras"] for r in vm + serverless})
    for ax in (done, time_ax):
        ax.set(xlabel="cameras", xscale="log", xticks=ticks, xticklabels=ticks)
        ax.minorticks_off()
        ax.grid(alpha=0.3)
        ax.legend()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def by_level(sent: list[dict], resources: list[dict], latency: list[dict], interval: float):
    """One summary row per number of cameras (the camera names n08-c03 encode the level)."""
    groups, latency_groups = defaultdict(list), defaultdict(list)
    for row in sent:
        groups[row["camera"].split("-")[0]].append(row)
    for row in latency:
        latency_groups[row["camera"].split("-")[0]].append(row)
    return [
        summarise(groups[level], resources, latency_groups[level], interval)
        for level in sorted(groups, key=lambda level: int(level[1:]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "load", "versus"))
    parser.add_argument("sent", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--plot", type=Path, help="graph file for load and versus modes")
    parser.add_argument("--other", type=Path, help="serverless sent.csv, for versus mode")
    parser.add_argument("--other-latency", type=Path, help="serverless counts.csv, for versus")
    args = parser.parse_args()

    sent, resources, latency = read(args.sent), read(args.resources), read(args.latency)
    if args.mode == "run":
        print(markdown([summarise(sent, resources, latency, args.interval)]))
        return
    rows = by_level(sent, resources, latency, args.interval)
    if args.mode == "versus":
        other = by_level(read(args.other), [], read(args.other_latency), args.interval)
        plot_versus(rows, other, args.plot)
        return
    print(markdown(rows))
    if args.plot:
        plot_levels(rows, args.plot)


if __name__ == "__main__":
    main()
