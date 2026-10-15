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
connection. `camera.capture()` calls it on a fixed schedule and yields timestamped frames; this
is the capture process that Parts 2 and 3 reuse. Design decisions:

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

`results.CsvLog` appends one row per frame to `counts.csv`, and `results.plot_counts` draws
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

**Verification.** Automated tests (`pytest`) check the processing loop, skipping of failed
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

In this part the laptop only captures frames; detection moves to a VM in Azure
(Infrastructure as a Service: we rent the machine and manage everything on it).

```text
laptop: sender.py ──JPEG + token──▶ VM: vm.py ─┬─ YOLOv8 → counts.csv, annotated frames
                                               ├─ monitor.py → resources.csv (CPU, RAM, network)
                                               └─ dashboard: histogram, time series, latest frame
```

### Task 2.1 — Creating and configuring the VM

| Setting | Value | Reason |
|---|---|---|
| Size | Standard_B1s: 1 vCPU, 1 GiB RAM | Recommended by the assignment; 750 free hours per month with Azure for Students |
| Image | Ubuntu Server 24.04 LTS | Long-term support; ships Python 3.12 |
| Region | *[region used]* | Close to the laptop, for low latency |
| Inbound ports | 22 (SSH), 80 (HTTP) | Administration and the public dashboard |
| Authentication | SSH key, no password | Cannot be guessed by brute force |

*[📸 VM overview and inbound port rules.]*

`deploy/setup.sh` makes the setup repeatable instead of a list of manual commands. It:

1. installs the required packages `python3-opencv` and `python3-pip` (plus `python3-venv`);
2. adds a 2 GiB swap file. With 1 GiB of RAM, installing PyTorch and loading the model can
   exhaust memory, and Linux then kills the process (the "killed" error in the assignment's
   note 2). Swap lets the system move rarely used memory to disk;
3. installs the CPU-only build of PyTorch and the project in a virtual environment, with
   `--no-cache-dir` (assignment note 2). Ubuntu 24.04 does not allow `pip` to install
   system-wide, and a virtual environment keeps the project isolated;
4. creates a random upload token in `/etc/smart-lighting.env`, readable only by root;
5. installs a **systemd** service. The service starts at boot, restarts on failure, and runs as
   a normal user rather than root. It is still allowed to use port 80
   (`CAP_NET_BIND_SERVICE`).

### Task 2.2 — Sending frames from the laptop

`sender.py` reuses the Task 1.2 capture loop (`camera.capture`). Each frame is compressed to
JPEG (quality 90, ≈110 kB for a 1280 × 720 frame) and sent in an HTTP `POST /frames` request.
The VM answers with the people count. The sender logs each answer together with the
**round-trip time** and the **bytes sent**; these are the data for Task 2.5.

Design decisions:

- **HTTP with a JPEG body.** HTTP is simple and passes through firewalls, and the same request
  works for any number of cameras in Part 4. JPEG is about 25× smaller than the raw frame.
- **Shared-secret token.** Port 80 is public because of the dashboard, so the upload endpoint
  requires an `X-API-Key` header. Without it, anyone on the internet could inject frames.
  (HTTPS would also encrypt the token; we left it out to keep the practice focused, since the
  frames come from a public camera.)

### Task 2.3 — Counting people on the VM

`vm.py` is a small **FastAPI** web service. For each uploaded frame it:

1. checks the token, and checks that the body is a valid image;
2. runs **the same `PersonDetector` as Part 1** (YOLOv8n, class *person*, confidence ≥ 0.5),
   so local and cloud results are directly comparable;
3. saves the annotated frame and appends a row to `counts.csv`, in the same format as Part 1.

The detector runs one frame at a time (a lock). The B1s has a single vCPU, so running frames in
parallel would not be faster and would use more memory. Extra frames wait in line, and that
waiting shows up in the response time. This is the effect we measure in Part 4.

*[📸 `counts.csv` on the VM.]*

### Task 2.4 — Results website

The same service serves a dashboard at `http://<public-ip>/`, refreshed every 5 seconds:

- a **histogram** of people per frame (how often 0, 1, 2 … people were seen);
- people over time, and the latest annotated frame;
- frames processed, mean people, mean inference time, and the VM's current CPU, memory and
  network use.

*[📸 dashboard opened through the public IP.]*

### Task 2.5 — Performance

*[To do after the Azure run: detection effectiveness, CPU, memory, bandwidth, cost and response
time, from `sent.csv`, `resources.csv` and Azure Monitor.]*

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
