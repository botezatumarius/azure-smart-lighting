# Practice 1: Cloud Computing Services — Project Report

## 1. Introduction

A Smart Lighting System (SLS) switches street lights on and off depending on whether people
or vehicles are present in a control zone. Cameras are a common presence sensor: each frame is
analysed by an AI model and the resulting count drives the lighting controller.

This assignment builds that sensing pipeline and moves the AI processing step by step to the
cloud, to observe the advantages and costs of each model:

1. **Local processing:** capture and detect on a laptop.
2. **Azure virtual machine (IaaS):** the laptop captures, a cloud VM detects.
3. **Serverless (FaaS) with Azure AI:** uploads to Blob Storage trigger an Azure Function that
   calls the Azure AI Vision service.
4. **Multiple cameras:** find how many cameras each cloud architecture can handle.

All parts use the same people count (YOLOv8 or Azure AI) and record results in the same CSV
format, so the architectures can be compared directly.

## 2. Part 1: Local processing

### Task 1.1 — Webcam selection

The teaching team's sample camera (`http://83.56.31.69/mjpg/video.mjpg`) no longer responds,
so we picked a replacement that meets the requirements:

- **Camera:** official municipal webcam of the Gaggenau market square (Germany),
  <https://www.gaggenau.de/webcams.212.htm>.
- **Fixed:** the camera does not pan or zoom, so the control zone is always the same.
- **Scene:** a pedestrian square with shops, comparable to the promenade in the assignment.
- **Access:** a public JPEG endpoint (`https://www.gaggenau.de/webcam/image00001.jpg`),
  1280 × 720 pixels, which the city refreshes every 5 seconds. This matches Task 1.2's rate.

*[Screenshot: a raw daytime frame from the camera.]*

### Task 1.2 — Capturing one frame every 5 seconds

`camera.read_frame()` uses OpenCV to open the camera URL, read one frame and close the
connection. Design decisions:

- **A new connection for each frame.** A stream that stays open buffers frames while the
  model is busy, so the next frame read would be seconds old. Reconnecting always returns the
  current image. At one frame per 5 s this costs little (≈0.5 s per read in our tests).
- **One code path for every camera type.** OpenCV (through FFmpeg) reads JPEG snapshots, MJPEG
  streams and HLS streams, so other cameras can be added in Part 4 by adding a single URL.
- **Timeouts.** A camera that does not respond fails after 10 s instead of blocking the program.
  That frame is logged and skipped, and capture continues. We do **not** record a count of 0,
  because a camera failure is not the same as an empty street.
- **A fixed schedule.** Captures are planned at t = 0, 5, 10 … s on a monotonic clock. A naive
  `sleep(5)` after each frame would add the processing time to every interval and make the
  timing drift.

Resolution: the camera delivers 1280 × 720. YOLOv8 resizes frames to 640 px internally, and at
that size people in the square are large enough to detect.

### Task 1.3 — Counting people with YOLOv8

`detector.PersonDetector` loads the pretrained **YOLOv8n** model (the smallest YOLOv8, suited to
CPU-only machines and later to a small VM). For each frame it:

1. runs YOLOv8 restricted to COCO class 0 (*person*) with confidence ≥ 0.5, the same threshold
   as the teaching team's sample;
2. counts the returned boxes;
3. returns the count, an image with the boxes drawn on it, and the inference time in ms.

The model is loaded once and warmed up with a blank image, so the first real frame's time is
not inflated by initialisation. The same class runs unchanged on the VM in Part 2.

`results.CountLog` appends one row per frame to `counts.csv`, and `results.plot_counts` draws
the time-series graph:

```csv
timestamp,camera,people,inference_ms
2026-10-15T19:39:55+02:00,gaggenau,0,148.9
2026-10-15T19:40:00+02:00,gaggenau,0,61.2
```

The timestamps include the UTC offset, so they remain unambiguous when compared with the cloud
VM (Part 2), which runs on UTC. The `camera` column lets the same file and graph hold several
cameras in Part 4.

*[Sample of `counts.csv`, two annotated frames and `people-over-time.png` from the daytime run.]*

**Verification.** Five automated tests (`pytest`) check the capture loop, skipping of failed
frames, the CSV format and the graph. They use a fake detector and camera, so they run in
seconds. We confirmed the detector on real footage with people (OpenCV's `vtest.avi`
pedestrian clip: 6 of 6 people found) and ran the full pipeline on the live camera.

### Problems encountered

| Problem | Solution |
|---|---|
| The sample camera from the teaching team is offline. | Picked an official, fixed municipal camera with a 5-second refresh. |
| At night the square is dark and empty, so every count was 0. | Recorded the evidence run during the day. |
| The sample code sleeps 0.5 s between frames and has no CSV header. | 5-second fixed schedule; CSV with a header. |

## 3. Part 2: Azure virtual machine

*To do.*

## 4. Part 3: Serverless and Azure AI

*To do.*

## 5. Part 4: Multi-camera management

*To do.*

## 6. Conclusions

*To do.*

## References

- Ultralytics YOLOv8: <https://docs.ultralytics.com/models/yolov8/>
- OpenCV `VideoCapture`: <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>
- Gaggenau webcams: <https://www.gaggenau.de/webcams.212.htm>
