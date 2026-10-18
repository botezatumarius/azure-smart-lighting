# Smart Lighting System

Practice 1 of **Architectures and Infrastructures for AI**: count people in public webcam
frames with YOLOv8, first locally and then on Azure.

All four parts are implemented and measured. The report is [docs/report.md](docs/report.md);
`python docs/build_pdf.py` renders it to `docs/report-Marius Botezatu.pdf` for submission.
The measurement data behind it is in [results/](results/README.md).

## Setup

Requires Python 3.11+. From the project folder (PowerShell):

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

## Part 3: serverless with Azure AI

The laptop uploads frames to the `raw` Blob container; Event Grid triggers the Azure Function
`ProcessFrame` (`functions/`), which counts people with Azure AI Vision, draws their boxes and
writes the results to the static website. Follow [docs/azure-serverless.md](docs/azure-serverless.md).

```powershell
$env:AZURE_STORAGE_CONNECTION_STRING = "<connection string>"
.venv\Scripts\python -m smart_lighting.uploader --publish-website   # once
.venv\Scripts\python -m smart_lighting.uploader --frames 240        # one frame every 0.5 s
```

## Part 4: multiple cameras

```powershell
.venv\Scripts\python -m smart_lighting.loadtest --target vm --server http://<public-ip> --levels 1 2 4 8 16 24 32
.venv\Scripts\python -m smart_lighting.loadtest --target blob --levels 1 4 16 32 64 128
```

Each simulated camera sends the real camera's latest frame every 5 s. The results are
summarised and plotted with `smart_lighting.analysis`; `smart_lighting.compare` runs YOLOv8 and
Azure AI on the same images to measure detection effectiveness.

## Advanced work

```powershell
.venv\Scripts\python -m smart_lighting.uploader --frames 240 --only-changes   # motion filter
.venv\Scripts\python -m smart_lighting.compare --sweep                       # resolution / JPEG quality
.venv\Scripts\python -m smart_lighting.tracking data/vtest.avi               # unique people
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
  streaming.py  send frames to a cloud target and log each result (Parts 2–4)
  uploader.py   Part 3: capture → upload to Blob Storage; publish the website
  loadtest.py   Part 4: simulate N cameras
  analysis.py   performance tables and graphs
  compare.py    YOLOv8 vs Azure AI on the same images
  tracking.py   advanced: count unique people by tracking (ByteTrack)
  web/          dashboard page (VM and static website)
functions/      Part 3: Azure Function (blob trigger → Azure AI Vision)
deploy/         VM setup script and systemd service
docs/           report, deployment docs, PDF build
results/        raw measurement data (CSV) behind the report
```
