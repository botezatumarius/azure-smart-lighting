"""Tasks 2.5 / 3.6: compare YOLOv8 and Azure AI Vision on the same camera images.

Downloads the distinct frames uploaded to the `raw` container during a run, counts people
with both detectors, and saves each image with both sets of boxes (YOLO blue, Azure green)
so that the true number of people can be counted by eye.

    python -m smart_lighting.compare --since 2026-10-18T13:29:30Z --until 2026-10-18T13:32:00Z
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from smart_lighting.detector import PersonDetector
from smart_lighting.results import CsvLog
from smart_lighting.uploader import RAW, connect

CONFIDENCE = 0.5
GREEN = (0, 200, 0)


def distinct_frames(since: datetime, until: datetime) -> list[tuple[str, bytes]]:
    """Frames uploaded in the time window, without repeats (the camera refreshes every 5 s)."""
    raw = connect().get_container_client(RAW)
    seen, frames = set(), []
    for blob in sorted(raw.list_blobs(), key=lambda b: b.name):
        digest = bytes(blob.content_settings.content_md5 or b"")
        if since <= blob.creation_time <= until and digest not in seen:
            seen.add(digest)
            frames.append((blob.name, raw.download_blob(blob.name).readall()))
    return frames


def azure_boxes(client: ImageAnalysisClient, jpeg: bytes) -> list:
    result = client.analyze(image_data=jpeg, visual_features=[VisualFeatures.PEOPLE])
    return [p.bounding_box for p in result.people.list if p.confidence > CONFIDENCE]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=datetime.fromisoformat, required=True)
    parser.add_argument("--until", type=datetime.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/compare"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log = CsvLog(args.output / "compare.csv", ("image", "yolo_people", "azure_people"))
    yolo = PersonDetector()
    vision = ImageAnalysisClient(
        os.environ["VISION_ENDPOINT"], AzureKeyCredential(os.environ["VISION_KEY"])
    )
    for index, (name, jpeg) in enumerate(distinct_frames(args.since, args.until), start=1):
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        detection, boxes = yolo.detect(frame), azure_boxes(vision, jpeg)
        image = detection.annotated
        for b in boxes:
            cv2.rectangle(image, (b.x, b.y), (b.x + b.width, b.y + b.height), GREEN, 2)
        cv2.imwrite(str(args.output / f"{index:02d}.jpg"), image)
        log.append(image=name, yolo_people=detection.people, azure_people=len(boxes))
        print(f"{index:02d} {name}: YOLO {detection.people}, Azure {len(boxes)}")


if __name__ == "__main__":
    main()
