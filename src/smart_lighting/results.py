"""Store results in CSV files and plot people counts over time."""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

COUNT_FIELDS = ("timestamp", "camera", "people", "inference_ms")


class CsvLog:
    """Append-only CSV file; the header is written once, when the file is created."""

    def __init__(self, path: Path, fields: tuple[str, ...]):
        self.path = path
        self.fields = fields
        if not path.exists():
            self._write(dict(zip(fields, fields, strict=True)), mode="w")

    def append(self, **row) -> None:
        self._write({key: _format(value) for key, value in row.items()})

    def _write(self, row: dict, mode: str = "a") -> None:
        with self.path.open(mode, newline="", encoding="utf-8") as file:
            csv.DictWriter(file, self.fields).writerow(row)


def _format(value):
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, float):
        return round(value, 1)
    return value


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
