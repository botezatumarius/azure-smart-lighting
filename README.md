# Smart Lighting System

Practice 1 of **Architectures and Infrastructures for AI**: count people in public webcam
frames with YOLOv8, first locally and then on Azure.

| Part | Status |
|---|---|
| 1 · Local processing | Done |
| 2 · Azure virtual machine | Code done; Azure deployment pending |
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

## Part 2: count people on an Azure VM

The laptop captures frames and sends them over HTTP to a service on the VM. The service counts
people with the same YOLOv8 detector, keeps the results in CSV files and serves a public
dashboard at `http://<public-ip>/`.

```text
laptop: sender.py ──JPEG + token──▶ VM: vm.py ─┬─ YOLOv8 → counts.csv, annotated frames
                                               ├─ monitor.py → resources.csv (CPU, RAM, network)
                                               └─ dashboard: histogram, time series, latest frame
```

Follow [docs/azure-vm.md](docs/azure-vm.md) to create the VM and deploy. Then:

```powershell
$env:SMART_LIGHTING_TOKEN = "<token>"
.venv\Scripts\python -m smart_lighting.sender --server http://<public-ip> --frames 60
```

The sender logs the round-trip time and upload size of each frame in `runs/sent-<camera>/sent.csv`.

To try the service on this computer, start it in one terminal and send frames from another
using `--server http://127.0.0.1:8000`:

```powershell
$env:SMART_LIGHTING_TOKEN = "test"; .venv\Scripts\python -m smart_lighting.vm
```

## Development

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests
```

## Layout

```text
src/smart_lighting/
  camera.py     camera list, 5-second capture loop (Task 1.2), JPEG encoding
  detector.py   YOLOv8 person detector, shared by laptop and VM
  results.py    CSV logs and time-series graph
  local.py      Part 1: capture → detect → save
  sender.py     Part 2: capture → send to VM
  vm.py         Part 2: VM web service (upload API + dashboard)
  monitor.py    Part 2: CPU, memory and network sampling on the VM
  web/          dashboard page
deploy/         VM setup script and systemd service
docs/           report draft and Azure guide
```
