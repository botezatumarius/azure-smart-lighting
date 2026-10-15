from datetime import datetime, timedelta

import pytest

from smart_lighting.results import COUNT_FIELDS, CsvLog, plot_counts


def test_csv_log_keeps_one_header_across_runs(tmp_path):
    path = tmp_path / "counts.csv"
    start = datetime.now().astimezone()
    CsvLog(path, COUNT_FIELDS).append(timestamp=start, camera="cam", people=3, inference_ms=41.27)
    CsvLog(path, COUNT_FIELDS).append(
        timestamp=start + timedelta(seconds=5), camera="cam", people=0, inference_ms=39.0
    )

    lines = path.read_text().splitlines()
    assert lines[0] == "timestamp,camera,people,inference_ms"
    assert lines[1].endswith(",cam,3,41.3")
    assert len(lines) == 3


def test_plot_draws_every_camera(tmp_path):
    path = tmp_path / "counts.csv"
    log, now = CsvLog(path, COUNT_FIELDS), datetime.now().astimezone()
    for camera in ("a", "b"):
        log.append(timestamp=now, camera=camera, people=1, inference_ms=10.0)

    plot_counts(path, tmp_path / "plot.png")

    assert (tmp_path / "plot.png").stat().st_size > 0


def test_plot_rejects_empty_csv(tmp_path):
    path = tmp_path / "counts.csv"
    CsvLog(path, COUNT_FIELDS)
    with pytest.raises(ValueError):
        plot_counts(path, tmp_path / "plot.png")
