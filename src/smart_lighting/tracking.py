"""Advanced work: count unique people passing a camera by tracking them across frames.

Counting people per frame counts the same person again in every frame they appear in.
A tracker (ByteTrack, built into Ultralytics) links detections across frames and gives each
person an ID, so the number of distinct IDs is the number of people who passed. The test
is repeated at lower frame rates to show how much frame rate tracking needs.

    python -m smart_lighting.tracking data/vtest.avi
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from smart_lighting.detector import PERSON


def count_people(video: Path, step: int, model: Path) -> tuple[int, int, int]:
    """Track every `step`-th frame. Returns (frames used, tracked boxes, unique IDs).

    At full frame rate the tracked boxes equal the per-frame detections; at low rates the
    tracker cannot confirm tracks and returns fewer.
    """
    tracker = YOLO(model)  # fresh tracker per run
    capture = cv2.VideoCapture(str(video))
    used = detections = 0
    ids: set[int] = set()
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            results = tracker.track(frame, persist=True, classes=[PERSON], conf=0.5, verbose=False)
            result = results[0]
            used += 1
            detections += len(result.boxes)
            if result.boxes.id is not None:
                ids.update(result.boxes.id.int().tolist())
        index += 1
    capture.release()
    return used, detections, len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"))
    args = parser.parse_args()

    fps = cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FPS)
    print("| Frames per second | Frames used | Tracked boxes | Unique people (IDs) |")
    print("|---|---|---|---|")
    for rate in (fps, fps / 2, fps / 5, 1.0, 0.2):
        used, detections, unique = count_people(args.video, max(1, round(fps / rate)), args.model)
        print(f"| {rate:g} | {used} | {detections} | {unique} |")


if __name__ == "__main__":
    main()
