"""Store people counts in a CSV file and plot them over time."""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

FIELDS = ("timestamp", "camera", "people", "inference_ms")


class CountLog:
    """Append-only CSV of people counts; one row per processed frame."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            self._write(FIELDS, mode="w")

    def append(self, timestamp: datetime, camera: str, people: int, inference_ms: float) -> None:
        self._write(
            (timestamp.isoformat(timespec="seconds"), camera, people, round(inference_ms, 1))
        )

    def _write(self, row: tuple, mode: str = "a") -> None:
        with self.path.open(mode, newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)


def plot_counts(csv_path: Path, output: Path) -> None:
    """Draw people per frame over time, one line per camera."""
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{csv_path} has no counts to plot")

    figure, axes = plt.subplots(figsize=(10, 4), layout="constrained")
    for camera in sorted({row["camera"] for row in rows}):
        series = [row for row in rows if row["camera"] == camera]
        times = [datetime.fromisoformat(row["timestamp"]) for row in series]
        axes.plot(times, [int(row["people"]) for row in series], marker="o", label=camera)

    axes.set(title="People detected per frame", xlabel="Time", ylabel="People")
    axes.set_ylim(0, max(int(row["people"]) for row in rows) + 1)
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=times[0].tzinfo))
    axes.grid(alpha=0.3)
    axes.legend()
    figure.savefig(output, dpi=150)
    plt.close(figure)
