"""Generate a standalone time-series figure directly from recorded CSV rows."""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def plot_counts(csv_path: Path, output: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The CSV contains no observations to plot.")
    is_video = all(row.get("video_time_s", "") != "" for row in rows)
    times = (
        [float(row["video_time_s"]) for row in rows]
        if is_video
        else [
            datetime.fromisoformat(row.get("timestamp_utc") or row["captured_utc"]) for row in rows
        ]
    )
    counts = [int(row["person_count"]) for row in rows]
    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(figsize=(10, 4.5), layout="constrained")
        axes.plot(times, counts, marker="o", markersize=3, color="#007f86", linewidth=1.5)
        axes.set(
            title="People detected per frame" + (" (recorded video)" if is_video else ""),
            xlabel="Position in recorded video (seconds)" if is_video else "Acquisition time (UTC)",
            ylabel="People",
        )
        axes.set_ylim(0, max(1, max(counts) + 1))
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        if not is_video:
            locator = mdates.AutoDateLocator()
            axes.xaxis.set_major_locator(locator)
            axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=times[0].tzinfo))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)
