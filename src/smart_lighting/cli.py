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
    root = argparse.ArgumentParser(description="Smart lighting: local and VM people counting")
    commands = root.add_subparsers(dest="command", required=True)
    local = commands.add_parser("local", help="Capture and count people, then save evidence")
    sender = commands.add_parser("send", help="Capture locally and send to a VM through SSH")
    sender.add_argument("--endpoint", default="http://127.0.0.1:8000")
    sender.add_argument(
        "--jpeg-quality", type=int, choices=range(1, 101), default=95, metavar="1-100"
    )
    sender.add_argument("--timeout", type=positive_float, default=30)
    for command in (local, sender):
        command.add_argument(
            "--source", required=True, help="Public webcam URL or local video path"
        )
        command.add_argument(
            "--source-kind", choices=("stream", "snapshot", "video"), default="stream"
        )
        command.add_argument("--camera-id", default="camera-01")
        command.add_argument("--frames", type=positive_int, default=60)
        command.add_argument("--interval", type=positive_float, default=5.0)
        command.add_argument("--output", type=Path)
    server = commands.add_parser("serve", help="Run the private VM detector and dashboard")
    server.add_argument("--port", type=positive_int, default=8000)
    server.add_argument("--threads", type=positive_int, default=1)
    server.add_argument("--output", type=Path)
    for command in (local, server):
        command.add_argument("--confidence", type=positive_float, default=0.5)
        command.add_argument("--image-size", type=positive_int, default=640)
        command.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"))
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
        elif args.command == "serve":
            if args.confidence > 1 or args.port > 65535:
                root.error("Confidence must be at most 1 and port at most 65535.")
            import uvicorn

            from smart_lighting.vm import VMConfig, create_app

            args.model.parent.mkdir(parents=True, exist_ok=True)
            output = args.output or Path("runs") / datetime.now(UTC).strftime("vm-%Y%m%dT%H%M%S%fZ")
            uvicorn.run(
                create_app(
                    VMConfig(output, args.model, args.confidence, args.image_size, args.threads)
                ),
                host="127.0.0.1",
                port=args.port,
                workers=1,
                limit_concurrency=8,
                proxy_headers=False,
            )
        else:
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
            output = args.output or Path("runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            if args.command == "send":
                from smart_lighting.sender import SendConfig, send

                send(
                    SendConfig(
                        args.source,
                        args.source_kind,
                        args.camera_id,
                        args.frames,
                        args.interval,
                        args.endpoint,
                        output,
                        args.jpeg_quality,
                        args.timeout,
                    )
                )
            else:
                if args.confidence > 1:
                    root.error("--confidence must be at most 1.")
                from smart_lighting.local import RunConfig, run

                args.model.parent.mkdir(parents=True, exist_ok=True)
                run(
                    RunConfig(
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
                )
            logging.info("Saved run to %s", output.resolve())
    except KeyboardInterrupt:
        logging.info("Stopped by user; completed observations were preserved.")
        return 130
    except Exception as error:
        logging.error("%s: %s", type(error).__name__, error)
        return 1
    return 0
