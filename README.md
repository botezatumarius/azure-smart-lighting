# Smart Lighting System

Practice 1 of **Architectures and Infrastructures for AI**: count people in public webcam
frames with YOLOv8, first locally and then on Azure.

| Part | Status |
|---|---|
| 1 · Local processing | Done |
| 2 · Azure virtual machine | To do |
| 3 · Serverless and Azure AI | To do |
| 4 · Multi-camera management | To do |

## Setup

Requires Python 3.10+. From the project folder (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[dev]"
```

The CPU build of PyTorch is enough and much smaller than the GPU build. YOLOv8 weights
(`models/yolov8n.pt`) download automatically on first use.

## Part 1: count people locally

```powershell
.venv\Scripts\python -m smart_lighting.local --frames 60
```

This takes one frame from the camera every 5 seconds, counts the people in it and writes to
`runs/<camera>/`:

| Output | Content |
|---|---|
| `counts.csv` | `timestamp, camera, people, inference_ms`, one row per frame |
| `frames/*.jpg` | Each frame with a box around every detected person |
| `people-over-time.png` | Graph of people per frame over time |

Omit `--frames` to run until **Ctrl+C**. Repeated runs append to the same CSV.
Cameras are listed in `CAMERAS` in `src/smart_lighting/camera.py`.

## Development

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests
```

## Layout

```text
src/smart_lighting/
  camera.py     camera list and single-frame capture
  detector.py   YOLOv8 person detector (reused on the VM in Part 2)
  results.py    CSV log and time-series graph
  local.py      Part 1: capture → detect → save loop
docs/report.md  project report draft
```
