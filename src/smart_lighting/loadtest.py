"""Part 4: simulate an increasing number of cameras, each sending one frame every 5 seconds.

The real camera is read once per interval and every simulated camera sends its latest frame.
That keeps the load on the public camera's server constant, while the cloud receives exactly
the traffic of N real cameras: N frames per interval, spread evenly over the interval.
"""

import argparse
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from smart_lighting.camera import CAMERAS, capture, read_frame
from smart_lighting.results import CsvLog
from smart_lighting.streaming import SENT_FIELDS, Target, stream

log = logging.getLogger(__name__)


class SharedCamera(threading.Thread):
    """Keeps the latest frame of one real camera, refreshed once per interval."""

    def __init__(self, url: str, interval: float):
        super().__init__(daemon=True)
        self.url = url
        self.interval = interval
        self.frame = read_frame(url)

    def run(self) -> None:
        for _, frame in capture(self.url, self.interval):
            self.frame = frame

    def read(self, _url: str) -> np.ndarray:
        return self.frame


def run_level(
    cameras: int,
    shared: SharedCamera,
    make_target: Callable[[], Target],
    sent: CsvLog,
    interval: float,
    duration: float,
) -> None:
    """Run `cameras` simulated cameras for `duration` seconds. Names encode the level: n08-c03."""

    def camera(index: int) -> None:
        time.sleep(index * interval / cameras)  # stagger cameras
        frames = capture("", interval, round(duration / interval), read=shared.read)
        stream(frames, make_target(), f"n{cameras:02d}-c{index:02d}", sent)

    with ThreadPoolExecutor(max_workers=cameras) as pool:
        list(pool.map(camera, range(cameras)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("vm", "blob"), required=True)
    parser.add_argument("--server", help="VM address, for --target vm")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--duration", type=float, default=120, help="seconds per level")
    parser.add_argument("--cooldown", type=float, default=30, help="pause between levels (s)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames")
    parser.add_argument("--camera", choices=CAMERAS, default="gaggenau")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.target == "vm":
        from smart_lighting.sender import VmTarget

        token = os.environ["SMART_LIGHTING_TOKEN"]

        def make_target():
            return VmTarget(args.server, token)
    else:
        from smart_lighting.uploader import BlobTarget, connect

        service = connect()

        def make_target():
            return BlobTarget(service)

    output = Path("runs") / f"load-{args.target}"
    output.mkdir(parents=True, exist_ok=True)
    sent = CsvLog(output / "sent.csv", SENT_FIELDS)
    shared = SharedCamera(CAMERAS[args.camera], args.interval)
    shared.start()
    for number, cameras in enumerate(args.levels):
        if number:
            time.sleep(args.cooldown)  # let the queue drain
        log.info("=== %d cameras for %.0f s ===", cameras, args.duration)
        run_level(cameras, shared, make_target, sent, args.interval, args.duration)
    log.info("Done; results in %s", sent.path)


if __name__ == "__main__":
    main()
