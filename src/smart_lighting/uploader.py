"""Part 3: capture frames locally and upload them to Blob Storage for serverless processing.

Frames go to the private `raw` container, which triggers the Azure Function. Results are
written by the function to the static website container `$web`, which triggers nothing.
"""

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient, ContentSettings

from smart_lighting.camera import CAMERAS, capture
from smart_lighting.results import COUNT_FIELDS, CsvLog
from smart_lighting.streaming import SENT_FIELDS, stream

RAW = "raw"
WEBSITE = "$web"
FUNCTION_FIELDS = (*COUNT_FIELDS, "latency_ms")  # counts.csv columns (function)
WEB_FILES = {"index.html": "text/html", "dashboard.js": "text/javascript"}
WEB = Path(__file__).with_name("web")

logging.getLogger("azure").setLevel(logging.WARNING)  # quiet SDK request logs


class BlobTarget:
    """Uploads frames as raw/<camera>/<UTC capture time>.jpg."""

    def __init__(self, service: BlobServiceClient):
        self.raw = service.get_container_client(RAW)
        self.settings = ContentSettings(content_type="image/jpeg")

    def submit(self, camera: str, timestamp: datetime, jpeg: bytes) -> dict:
        # Name encodes camera and capture time.
        name = f"{camera}/{timestamp.astimezone(UTC):%Y%m%dT%H%M%S%fZ}.jpg"
        try:
            self.raw.upload_blob(name, jpeg, content_settings=self.settings)
        except AzureError as error:
            raise ConnectionError(f"Upload to Blob Storage failed: {error}") from error
        return {}


def publish_website(service: BlobServiceClient) -> None:
    """Upload the dashboard to $web and create the results CSV the function appends to."""
    website = service.get_container_client(WEBSITE)
    for name, content_type in WEB_FILES.items():
        website.upload_blob(
            name,
            (WEB / name).read_bytes(),
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    counts = website.get_blob_client("counts.csv")
    if not counts.exists():
        # Append blob: safe for concurrent writers.
        counts.create_append_blob(
            content_settings=ContentSettings(content_type="text/csv", cache_control="no-store")
        )
        counts.append_block(",".join(FUNCTION_FIELDS) + "\r\n")


def connect() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-website", action="store_true", help="upload the dashboard")
    parser.add_argument("--camera", choices=CAMERAS, default="gaggenau")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between frames")
    parser.add_argument("--frames", type=int, help="stop after N frames (default: until Ctrl+C)")
    parser.add_argument("--output", type=Path, help="log folder (default: runs/upload-<camera>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    service = connect()
    if args.publish_website:
        publish_website(service)
        logging.info("Website published")
        return
    output = args.output or Path("runs") / f"upload-{args.camera}"
    output.mkdir(parents=True, exist_ok=True)
    sent = CsvLog(output / "sent.csv", SENT_FIELDS)
    try:
        stream(
            capture(CAMERAS[args.camera], args.interval, args.frames),
            BlobTarget(service),
            args.camera,
            sent,
        )
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()
