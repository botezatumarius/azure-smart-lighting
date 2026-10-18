# Measurement data

Raw CSV logs behind the tables and graphs in the report. Timestamps include their UTC offset:
local files (laptop) use CEST (+02:00), files written on Azure use UTC. All times in milliseconds,
sizes in bytes. `python -m smart_lighting.analysis` reproduces the report's tables from these files.

| File | Report | Content |
|---|---|---|
| `part1/counts-1280px.csv` | Task 1.3 | daylight run, 16 Oct, detection at 1280 px (59 frames) |
| `part1/counts-640px.csv` | Task 1.2 | the same camera at YOLO's default 640 px (30 frames) |
| `part2/sent.csv` | Tasks 2.2, 2.5 | laptop side: VM answer, round-trip time, upload size per frame |
| `part2/vm-counts.csv` | Tasks 2.3, 4.2 | counts written on the VM: the Part 2 run (16 Oct 13:02–13:07 UTC), an interrupted first load-test attempt (16 Oct 14:05–14:08 UTC, not used) and the load test (17 Oct 12:17–12:49 UTC) |
| `part2/vm-resources.csv` | Tasks 2.5, 4.2 | VM CPU, memory and network, sampled every 5 s |
| `part3/sent.csv` | Task 3.6 | upload time and size per frame (one frame every 0.5 s) |
| `part3/function-counts.csv` | Task 3.6 | results written by the Azure Function: people, AI Vision time, latency |
| `part3/detector-comparison.csv` | Task 3.6 | YOLOv8 and Azure AI counts on the same 36 distinct images |
| `part3/ground-truth.csv` | Task 3.6 | people counted by eye on 12 of those images |
| `part4/vm-sent.csv` | Task 4.2 | VM load test, laptop side; camera `n08-c03` = camera 3 of the 8-camera step |
| `part4/serverless-sent.csv` | Task 4.3 | serverless load test, laptop side |
| `part4/serverless-function-counts.csv` | Task 4.3 | all function results (Part 3 run and load test) |
| `advanced/change-filter-sent.csv` | Section 6.1 | live upload run with the change filter |
| `advanced/resolution-sweep.csv` | Section 6.2 | people found and upload size per width and JPEG quality |

Example: the Task 4.2 table is

```powershell
python -m smart_lighting.analysis load results/part4/vm-sent.csv --resources results/part2/vm-resources.csv
```
