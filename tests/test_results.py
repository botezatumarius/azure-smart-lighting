from datetime import datetime, timedelta

import pytest

from smart_lighting.results import CountLog, plot_counts


def test_count_log_keeps_one_header_across_runs(tmp_path):
    path = tmp_path / "counts.csv"
    start = datetime.now().astimezone()
    CountLog(path).append(start, "cam", 3, 41.27)
    CountLog(path).append(start + timedelta(seconds=5), "cam", 0, 39.0)

    lines = path.read_text().splitlines()
    assert lines[0] == "timestamp,camera,people,inference_ms"
    assert lines[1].endswith(",cam,3,41.3")
    assert len(lines) == 3


def test_plot_draws_every_camera(tmp_path):
    path = tmp_path / "counts.csv"
    log, now = CountLog(path), datetime.now().astimezone()
    for camera in ("a", "b"):
        log.append(now, camera, 1, 10)

    plot_counts(path, tmp_path / "plot.png")

    assert (tmp_path / "plot.png").stat().st_size > 0


def test_plot_rejects_empty_csv(tmp_path):
    path = tmp_path / "counts.csv"
    CountLog(path)
    with pytest.raises(ValueError):
        plot_counts(path, tmp_path / "plot.png")
