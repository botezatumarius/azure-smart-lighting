# Smart Lighting System

Practical assignment for **Architectures and Infrastructures in AI**.
Part 1 is implemented. Part 2's sender, VM service and dashboard are implemented and tested locally;
Azure deployment and cloud measurements are pending. Parts 3-4 have not been started.

The pipeline is small: **public webcam → capture every 5 seconds → YOLOv8 → CSV + images + graph**.
It counts people visible in each frame. It does not count unique visitors across frames.

## Start here

1. Read [the assignment analysis](docs/assignment-analysis.md).
2. Follow [the Part 1 walkthrough](docs/part1-walkthrough.md).
3. Follow [the Part 2 walkthrough](docs/part2-walkthrough.md) for Azure VM creation and testing.
4. Maintain [the working report](docs/report.md) as experiments are completed.

The supplied PDFs and ZIP are preserved unchanged. The implementation follows the teaching team's
`grabber.py` sequence, with corrected timing, CSV headers and explicit experiment records.

## Test with a recorded video

The project supports repeatable offline testing with OpenCV's pedestrian sample
[`vtest.avi`](https://github.com/opencv/opencv/blob/4.12.0/samples/data/vtest.avi).
The video is already downloaded on this computer. On a fresh checkout, download and verify it first:

```powershell
& .\.venv\Scripts\python.exe scripts/download_demo.py
& .\.venv\Scripts\python.exe -m smart_lighting local --source data/vtest.avi --source-kind video --camera-id opencv-test
```

This uses the same detector and evidence pipeline as the webcam mode. It samples the video at
0, 5, 10 seconds and so on, processing offline without waiting five real seconds per frame.
It stops normally at the end of the clip or the requested `--frames` limit, whichever comes first.

The verified run in `runs/part1-recorded-video/` contains **16 sampled frames**, with **3-8 people
detected per frame**, a CSV, bounding-box images and a graph. Its manifest labels the source as
`recorded_video_test` and records the source checksum. The graph uses seconds within the video;
the separate UTC timestamps describe when this computer read each sample. These results demonstrate
positive detections on recorded footage; live-camera assignment evidence remains separate.

To test your own recording, replace `data/vtest.avi` with its file path, quoted if it contains spaces.
Downloaded recordings and generated outputs are excluded from Git. The demo downloader pins the
OpenCV 4.12.0 file and checks its SHA-256 before saving it.

## Run Part 1 on this computer

The project environment is already installed in `.venv`. Open a PowerShell terminal in this folder:

```powershell
& .\.venv\Scripts\python.exe -m smart_lighting local --source "https://www.gaggenau.de/webcam/image00001.jpg" --source-kind snapshot --camera-id gaggenau-market --frames 60
```

This collects 60 observations over approximately five minutes. Choose a daylight period with
visible pedestrians for the main report. The camera is a fixed municipal view of a pedestrian
market square; its [official page](https://www.gaggenau.de/webcams.212.htm) states a five-second refresh.
Source availability and lighting can change. A zero count means no people were **detected**, and is
not proof that nobody was present.

The supplied ZIP's webcam can be retried with:

```powershell
& .\.venv\Scripts\python.exe -m smart_lighting local --source "http://83.56.31.69/mjpg/video.mjpg" --frames 60
```

That endpoint timed out during setup on 12 October 2026. It is not the working default.

Each run has its own directory under `runs/`:

| Output | Purpose |
|---|---|
| `counts.csv` | UTC acquisition times, counts, resolution, timings, hashes and optional video positions |
| `people-over-time.png` | Report-ready time-series graph |
| `raw/*.png` | Losslessly saved decoded input frames for inspection and later comparison |
| `annotated/*.jpg` | People bounding boxes and confidence scores |
| `detections.jsonl` | Coordinates and confidence of every detected person |
| `run.json` | Settings, versions, model hash and completion/failure status |

Stop with **Ctrl+C** to preserve completed observations and generate a partial graph.
Failed captures are errors, not zero-person observations. Existing run folders cannot be overwritten.

## Install on another machine

Use Python 3.12 and run these commands from the project folder. CPU PyTorch avoids a large GPU
installation and provides a useful baseline for the later CPU VM comparison.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -e ".[vm,dev]"
```

`models/yolov8n.pt` downloads from Ultralytics on the first run. Model loading and a warm-up inference
finish before scheduled capture begins. No Azure subscription or API key is needed for Part 1.
`requirements-lock.windows.txt` records the tested Windows/Python 3.12 environment; use the CPU
PyTorch index above when recreating that environment. Create a separate lock for the Linux VM.

## Verify and regenerate a graph

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests scripts
& .\.venv\Scripts\python.exe -m smart_lighting plot runs/part1-gaggenau-evening-retry/counts.csv
```

Tests verify scheduling, zero-count recording, evidence creation, capture-failure handling and
protection against overwriting an experiment. They also check recorded-video sampling, normal
end-of-video handling, HTTP uploads, duplicate rejection, invalid images/metadata, overload rejection
and transport failures. They do not measure accuracy.

## Preview Part 2 on this computer

Start the detector in one PowerShell terminal:

```powershell
& .\.venv\Scripts\python.exe -m smart_lighting serve
```

In a second terminal, send three frames from the recorded video:

```powershell
& .\.venv\Scripts\python.exe -m smart_lighting send --source data/vtest.avi --source-kind video --camera-id opencv-test --frames 3
```

Open <http://127.0.0.1:8000/> to see the dashboard. Frames are sent five seconds apart, so this test
takes approximately ten seconds after the service is ready. Stop the server with Ctrl+C.
For an Azure deployment, the same commands run on separate machines and uploads go through SSH.
Follow the [Part 2 walkthrough](docs/part2-walkthrough.md) for the exact steps and public webpage.

The server saves `counts.csv`, `resources.csv`, received raw JPEGs, annotated images, detection JSON
and `run.json` in `runs/vm-.../`. The sender saves `client.csv`, raw PNGs and its own `run.json` in a
separate local run directory. `scripts/analyze_vm.py VM_RUN CLIENT_RUN` checks matching evidence,
summarises measurements and prepares a manual-count template. Generated evidence stays out of Git.

The completed evening verification has 60 observations, a mean acquisition interval of 5.001 s,
and a mean detector-call duration of 69.683 ms. All counts were zero in the low-light scene;
58 decoded images were distinct. See the working report for interpretation and limitations.
The separate static test on the assignment's example image detected five people.

## Code layout

```text
src/smart_lighting/
  capture.py     one image from a stream or snapshot endpoint
  detection.py   reusable YOLOv8 person detector
  local.py       scheduling and evidence collection
  scheduling.py  shared cadence calculation
  sender.py      local capture and private HTTP uploads
  vm.py          VM detector, persistence and read API
  metrics.py     CPU, memory and host network sampling
  web/           dashboard, time series and count histogram
  plotting.py    CSV-to-graph rendering
  cli.py         command-line arguments and errors
```

`deploy/` contains the Nginx configuration and Linux service. VM dependencies are installed with
the `vm` extra. No serverless infrastructure is included before Part 3.
