"""Small command-line interface; no credentials or camera addresses in the code."""

import argparse
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive, finite number.")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Smart lighting: Part 1 local people counting")
    commands = root.add_subparsers(dest="command", required=True)
    local = commands.add_parser("local", help="Capture and count people, then save evidence")
    local.add_argument("--source", required=True, help="Public webcam URL or local video path")
    local.add_argument("--source-kind", choices=("stream", "snapshot", "video"), default="stream")
    local.add_argument("--camera-id", default="camera-01")
    local.add_argument("--frames", type=positive_int, default=60)
    local.add_argument("--interval", type=positive_float, default=5.0)
    local.add_argument("--confidence", type=positive_float, default=0.5)
    local.add_argument("--image-size", type=positive_int, default=640)
    local.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"))
    local.add_argument("--output", type=Path)
    plot = commands.add_parser("plot", help="Regenerate the time-series figure from a run CSV")
    plot.add_argument("csv", type=Path)
    plot.add_argument("--output", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "plot":
            from smart_lighting.plotting import plot_counts

            plot_counts(args.csv, args.output or args.csv.with_name("people-over-time.png"))
        else:
            if args.confidence > 1:
                root.error("--confidence must be at most 1.")
            if args.source_kind == "video":
                video = Path(args.source).resolve()
                if not video.is_file():
                    root.error(
                        "--source must be an existing local video file for --source-kind video."
                    )
                args.source = str(video)
            else:
                source = urlsplit(args.source)
                if source.scheme not in {"http", "https", "rtsp"} or not source.hostname:
                    root.error("--source must be a direct HTTP, HTTPS or RTSP webcam URL.")
                if source.username or source.password:
                    root.error("Use a public webcam URL without embedded credentials.")
            from smart_lighting.local import RunConfig, run

            args.model.parent.mkdir(parents=True, exist_ok=True)
            output = args.output or Path("runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            config = RunConfig(
                args.source,
                args.source_kind,
                args.camera_id,
                args.frames,
                args.interval,
                args.confidence,
                args.image_size,
                args.model,
                output,
            )
            run(config)
            logging.info("Saved run to %s", output.resolve())
    except KeyboardInterrupt:
        logging.info("Stopped by user; completed observations were preserved.")
        return 130
    except Exception as error:
        logging.error("%s: %s", type(error).__name__, error)
        return 1
    return 0
