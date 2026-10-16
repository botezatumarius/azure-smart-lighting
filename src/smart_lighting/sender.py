"""Part 2: capture frames locally and send them to the Azure VM for people counting."""

import argparse
import contextlib
import logging
import os
from datetime import datetime
from pathlib import Path

import requests

from smart_lighting.camera import CAMERAS, capture
from smart_lighting.results import CsvLog, plot_counts
from smart_lighting.streaming import SENT_FIELDS, stream


class VmTarget:
    """POSTs JPEG frames to the VM service, which answers with the people count."""

    def __init__(self, server: str, token: str):
        self.url = f"{server.rstrip('/')}/frames"
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": token, "Content-Type": "image/jpeg"})

    def submit(self, camera: str, timestamp: datetime, jpeg: bytes) -> dict:
        try:
            response = self.session.post(self.url, params={"camera": camera}, data=jpeg, timeout=60)
            response.raise_for_status()
        except requests.RequestException as error:
            raise ConnectionError(f"Upload to VM failed: {error}") from error
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="VM address, e.g. http://20.1.2.3")
    parser.add_argument("--camera", choices=CAMERAS, default="gaggenau")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames")
    parser.add_argument("--frames", type=int, help="stop after N frames (default: until Ctrl+C)")
    parser.add_argument("--output", type=Path, help="log folder (default: runs/sent-<camera>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    output = args.output or Path("runs") / f"sent-{args.camera}"
    output.mkdir(parents=True, exist_ok=True)
    sent = CsvLog(output / "sent.csv", SENT_FIELDS)
    target = VmTarget(args.server, os.environ["SMART_LIGHTING_TOKEN"])
    try:
        stream(capture(CAMERAS[args.camera], args.interval, args.frames), target, args.camera, sent)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    with contextlib.suppress(ValueError):  # no frames sent
        plot_counts(sent.path, output / "people-over-time.png")


if __name__ == "__main__":
    main()
