# Smart Lighting System

Practical assignment for **Architectures and Infrastructures in AI**.
Part 1 is implemented; cloud resources and Parts 2-4 have not been created.

The pipeline is small: **public webcam → capture every 5 seconds → YOLOv8 → CSV + images + graph**.
It counts people visible in each frame. It does not count unique visitors across frames.

## Start here

1. Read [the assignment analysis](docs/assignment-analysis.md).
2. Follow [the Part 1 walkthrough](docs/part1-walkthrough.md).
3. Maintain [the working report](docs/report.md) as experiments are completed.

The supplied PDFs and ZIP are preserved unchanged. The implementation follows the teaching team's
`grabber.py` sequence, with corrected timing, CSV headers and explicit experiment records.

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
| `counts.csv` | UTC acquisition times, counts, resolution, timings and image hashes |
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
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`models/yolov8n.pt` downloads from Ultralytics on the first run. Model loading and a warm-up inference
finish before scheduled capture begins. No Azure subscription or API key is needed for Part 1.
`requirements-lock.windows.txt` records the tested Windows/Python 3.12 environment; use the CPU
PyTorch index above when recreating that environment. Create a separate lock for the Linux VM.

## Verify and regenerate a graph

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests
& .\.venv\Scripts\python.exe -m smart_lighting plot runs/part1-gaggenau-evening-retry/counts.csv
```

Tests verify scheduling, zero-count recording, evidence creation, capture-failure handling and
protection against overwriting an experiment. They do not measure model accuracy.

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
  plotting.py    CSV-to-graph rendering
  cli.py         command-line arguments and errors
```

No cloud dependencies or speculative framework are introduced before the corresponding part.
