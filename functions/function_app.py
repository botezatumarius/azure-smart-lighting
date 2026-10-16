"""Part 3: count people in each uploaded frame with Azure AI Vision.

Runs only for blobs created in the `raw` container (blob trigger + Event Grid filter).
Results go to the static website container `$web`, which never triggers this function,
so the function cannot call itself recursively.
"""

import io
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import PurePosixPath

import azure.functions as func
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from PIL import Image, ImageDraw

CONFIDENCE = 0.5  # same as YOLO
JPEG = ContentSettings(content_type="image/jpeg")
JPEG_NO_CACHE = ContentSettings(content_type="image/jpeg", cache_control="no-store")

app = func.FunctionApp()
vision = ImageAnalysisClient(
    os.environ["VISION_ENDPOINT"], AzureKeyCredential(os.environ["VISION_KEY"])
)
# Managed identity, no storage keys.
website = BlobServiceClient(
    os.environ["DataStorage__blobServiceUri"],  # noqa: SIM112
    DefaultAzureCredential(),
).get_container_client("$web")


@app.function_name(name="ProcessFrame")
@app.blob_trigger(arg_name="blob", path="raw/{name}", connection="DataStorage", source="EventGrid")
def process_frame(blob: func.InputStream) -> None:
    image = blob.read()
    started = time.perf_counter()
    result = vision.analyze(image_data=image, visual_features=[VisualFeatures.PEOPLE])
    vision_ms = (time.perf_counter() - started) * 1000
    boxes = [p.bounding_box for p in result.people.list if p.confidence > CONFIDENCE]

    # raw/<camera>/<UTC capture time>.jpg
    path = PurePosixPath(blob.name)
    camera = path.parent.name
    captured = datetime.strptime(path.stem, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)

    annotated = draw_boxes(image, boxes)
    website.upload_blob(
        f"frames/{camera}/{path.name}", annotated, overwrite=True, content_settings=JPEG
    )
    website.upload_blob("latest.jpg", annotated, overwrite=True, content_settings=JPEG_NO_CACHE)

    latency_ms = (datetime.now(UTC) - captured).total_seconds() * 1000
    row = (captured.isoformat(timespec="seconds"), camera, len(boxes), vision_ms, latency_ms)
    website.get_blob_client("counts.csv").append_block(
        ",".join(f"{value:.1f}" if isinstance(value, float) else str(value) for value in row)
        + "\r\n"
    )
    logging.info(
        "%s: %d people, vision %.0f ms, latency %.0f ms",
        blob.name,
        len(boxes),
        vision_ms,
        latency_ms,
    )


def draw_boxes(jpeg: bytes, boxes) -> bytes:
    """Draw a rectangle around each detected person, using the service's bounding_box."""
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(
            (box.x, box.y, box.x + box.width, box.y + box.height), outline="lime", width=3
        )
    output = io.BytesIO()
    image.save(output, "JPEG", quality=90)
    return output.getvalue()
