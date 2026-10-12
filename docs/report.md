# Smart Lighting System: Cloud Computing Services

**Working report - not yet a complete submission.**

Course: Architectures and Infrastructures in AI

Authors: To be completed by the student/group

Date: 13 October 2026

## Introduction

Smart lighting can adapt to the presence of people in a monitored urban area. This practical
implements the image-processing pipeline for such a system and compares three deployment
approaches: local processing, an Azure virtual machine and Azure serverless processing using an
Azure AI service. A final experiment increases the number of cameras to investigate capacity.

The objectives are to establish a working people-detection pipeline, move its processing to cloud
infrastructure, measure the resulting performance and cost, and explain the observed trade-offs
using reproducible evidence. The implementation counts visible people per frame; unique-person
passage counting is a separate optional extension.

## Task 1.1 - Public webcam

The supplied teaching-team script refers to `http://83.56.31.69/mjpg/video.mjpg`. That address timed
out during setup on 12 October 2026. We therefore selected the fixed municipal webcam overlooking
the pedestrian market square in Gaggenau, Germany, published on the city's official website [3].
The camera supplies a 1280 × 720 JPEG image, and the operator states a five-second refresh interval.
This provides an urban pedestrian control zone comparable to the promenade example in the brief.

Evidence is stored in the raw-frame directory of each run. The initial evening experiment has
low-light limitations. A daylight experiment with clearly visible pedestrians remains to be added
before the report is finalised.

## Task 1.2 - Local frame capture

The supplied example was used as the basis of the capture-detect-save workflow. Its 0.5-second
delay was changed to the required five-second cadence. A monotonic clock schedules each capture,
avoiding an extra inference-time delay on every interval. If processing exceeds the interval, the
program records the actual observation times and warns about skipped slots.

The program fetches the current webcam JPEG at each scheduled time and preserves the decoded
input as a lossless PNG. Timestamps are timezone-aware UTC values recorded at local acquisition
completion; they are not camera exposure timestamps. Input-resolution, timing and image hashes
are included in the CSV. Consecutive duplicate images produce a warning to assist freshness checks.

## Task 1.3 - Local people counting, CSV and graph

We use Ultralytics' pretrained YOLOv8n model [4] on the CPU. The model is loaded once and warmed up
before collection. Inference requests class 0 (person) with a confidence threshold of 0.5 and a
640-pixel inference-size setting. Returned person boxes are counted and drawn on the original image.

Each experiment writes `counts.csv` with a header, the acquisition timestamp and person count,
along with camera/frame identifiers and measurement metadata. It also saves annotated images,
per-detection coordinates/confidences, a time-series graph, and a run manifest containing settings,
library versions, completion status and a model checksum. Separate run folders prevent data from
different experiments being mixed. Capture failures are not represented as zero counts.

### Initial results

The completed live experiment is stored in `runs/part1-gaggenau-evening-retry/`. It ran on
12 October 2026 from 17:55:15.990 to 18:00:11.044 UTC (19:55-20:00 local time in Warsaw/Gaggenau).
The model/settings were YOLOv8n, CPU execution, confidence 0.5 and inference size 640.
Model initialisation and warm-up preceded scheduled collection.

| Measurement | Observed result |
|---|---:|
| Successfully recorded frames | 60 |
| Original decoded resolution | 1280 × 720 |
| First-to-last observation span | 295.054 s |
| Mean acquisition-completion interval | 5.001 s |
| Minimum / maximum acquisition-completion interval | 4.319 / 5.627 s |
| Mean detector-call duration | 69.683 ms |
| Mean capture duration (including retries) | 329.517 ms |
| Maximum processing duration (definition in manifest) | 1110.000 ms |
| Maximum scheduled-start lag | 16.000 ms |
| Distinct decoded-image hashes | 58 |
| Consecutive repeated images | 2 |
| Detected people per frame | 0 throughout |

These values were calculated from all 60 CSV rows; the summary is saved as
`measurement-summary.json`. Acquisition-completion intervals vary with network/read/retry duration,
while capture starts stay close to the intended five-second schedule. The two repeated images show
that separate requests do not guarantee a newly exposed camera frame on every request.

![People detected over time in the evening verification run](../runs/part1-gaggenau-evening-retry/people-over-time.png)

The graph is flat because the detector returned zero people in every frame. This verifies the
capture/inference/CSV/plot pipeline under the observed evening conditions; it does not establish
that the square was empty or that detection was accurate. A representative daylight run and manual
review remain necessary for stronger evidence of effectiveness. There is no basis here to claim
unique-person totals or to compare local and cloud performance yet.

CSV sample (the full file also includes identifiers, resolution, timings and image hashes):

| UTC acquisition timestamp | Detected people |
|---|---:|
| 2026-10-12T17:55:15.990+00:00 | 0 |

The paired raw and annotated images for this first observation are `raw/000001.png` and
`annotated/000001.jpg` in the experiment folder. They retain the original dimensions for visual review.

A separate static smoke test on the unannotated webcam image embedded in the assignment
(page 2, Figure 2) returned five person detections at confidence threshold 0.5. Its outputs are in
`runs/validation-assignment-image/`. This verifies positive detections on the provided example;
it is not a benchmark, a live sequence, or an independently labelled accuracy estimate.

### Recorded-video verification

To verify positive detections without depending on webcam lighting and availability, we added an
offline video input to the same pipeline. The test clip is OpenCV's `vtest.avi`, pinned to the 4.12.0
repository version [7], with SHA-256
`45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`.
It contains 795 frames at 10 fps (79.5 seconds), with a decoded resolution of 768 × 576.

We sampled at video positions 0, 5, 10, ..., 75 seconds, using the same YOLOv8n CPU detector,
confidence threshold 0.5 and inference size 640. The clip ended normally after 16 observations.
The resulting counts, in order, were:

`3, 5, 6, 7, 7, 7, 6, 5, 3, 3, 4, 5, 6, 6, 8, 7`.

![People detected over recorded video time](../runs/part1-recorded-video/people-over-time.png)

![Three detected people in the first recorded-video sample](../runs/part1-recorded-video/annotated/000001.jpg)

The first sample contains three clearly visible people, and the detector drew one box around each.
This is a visual sanity check, not a dataset-wide accuracy measurement. Later frames contain more
people, and the detected counts change over time. We have not independently labelled every sample.

The CSV's `video_time_s` column records each requested sampling position. The graph uses that media
timeline, while `timestamp_utc` records local acquisition during offline processing. No live-schedule
lag is reported for the video test. The manifest identifies the input as `recorded_video_test` and
records its source hash. These outputs are kept in `runs/part1-recorded-video/`, separate from live
webcam evidence. A downloader script reproduces and verifies the test input without committing the
video file to Git.

### Design decisions and problems

- The example webcam was unreachable, requiring a documented replacement.
- The first live run stopped after 15 observations when the snapshot endpoint returned an invalid image.
  Completed data and a partial graph were preserved. We added two bounded retries for transient camera
  read/decode failures and repeated the experiment in a separate directory.
- A JPEG snapshot endpoint avoids transferring a continuous video when only one frame/5 s is needed.
- The original JavaScript's fixed timezone offset was replaced with explicit UTC timestamps.
- Original decoded frames are retained to allow visual review and later comparison on the same inputs.
- Small, separately responsible modules support reuse of the detector on the VM without duplication.
- Low light and small/distant pedestrians can reduce effectiveness; manual review and daylight evidence
  are required before making claims about detection quality.

## Task 2.1 - Azure VM configuration

**Deployment pending.** The VM service, Linux service definition and Nginx configuration are prepared
in the repository. The student has not yet created an Azure VM. The deployment instructions are in
`docs/part2-walkthrough.md`; Ubuntu 24.04 LTS x64 provides the required Python 3.12 environment.
The service runs without root privileges and uses CPU-only PyTorch, one inference thread and one
model instance. Actual Linux dependency versions must be captured after installation.

The brief recommends Standard B1s, with 1 vCPU and 1 GiB RAM. Microsoft's current size documentation
marks the Bv1 family as End of Life [8]. We must check student-subscription regions, size availability,
remaining credit and displayed price before deployment. An affordable permitted current x64 size,
such as B2ls v2 with 2 vCPUs and 4 GiB [9], may be substituted with the reason documented. No VM size,
Azure cost or successful cloud deployment is claimed at this stage.

## Task 2.2 - Local capture and image transport

The `send` command reuses the Part 1 capture functions and cadence calculation, capturing locally
and sending one JPEG every five seconds to the VM. YOLO does not run in the sender. JPEG quality
defaults to 95, original image dimensions are retained, and the VM rejects uploads above 2 MB or
4096 pixels in either dimension. Locally decoded inputs are also preserved as PNGs for inspection.

HTTP upload requests travel inside an encrypted SSH tunnel. The inference service binds only to the
VM's loopback address. The public web proxy cannot forward uploads. Frame IDs, acquisition UTC,
camera ID, input type and optional recorded-video position accompany each upload. Client and VM
CSV records share IDs and JPEG checksums, permitting verification of the exact received input.
Acquisition UTC denotes capture completion, not camera exposure time.

The sender schedules recorded-video replays in real time, unlike Part 1's offline video test.
Missed capture slots are skipped and logged. Upload failures stop the client, preserve completed
records and write an error row without a person count. Uploads are not automatically retried because
a timeout may occur after the VM has already persisted the result.

## Task 2.3 - Remote detection and CSV persistence

The VM service calls the same `PersonDetector` used in Part 1: YOLOv8n, person class only, confidence
0.5 and inference size 640 on CPU. It loads and warms the model before accepting requests.
Ultralytics resets the CPU thread setting during backend initialisation, so the service applies the
requested thread count after initialisation and in the inference worker. The manifest records it.

The service preserves received JPEGs, annotated images and per-frame bounding-box/confidence JSON,
and writes UTC timestamps, counts, dimensions, hashes, payload size and processing durations to
`counts.csv`. One inference request and writer hold a lock at a time. Concurrent excess work is
rejected with HTTP 429; duplicate frame IDs are rejected with HTTP 409. Invalid image data and
metadata are not recorded as zero detections. Each service start creates a distinct experiment folder.

## Task 2.4 - Results webpage

The dashboard shows the latest person count, received frame count, annotated image, count histogram,
time series, host CPU and process memory. Recorded inputs are explicitly identified. Charts show
the latest 300 frames, while the downloadable CSV retains all observations in the service run.
The page uses local assets without external JavaScript/chart dependencies and refreshes every five
seconds. Refresh failures leave previously received values visible with an error message.

The supplied Nginx configuration publishes only named read endpoints on port 80. The intended Azure
rules expose HTTP 80 for the public-IP webpage and restrict SSH 22 to the student's IP. Inference
port 8000 remains closed publicly. Browser rendering and refresh were verified locally; Nginx and
public Azure IP access remain to be verified on the VM.

![Local dashboard preview, not an Azure deployment](../runs/part2-local-client-verified/dashboard.png)

## Task 2.5 - Measurements and local verification

Resource monitoring samples process CPU/RSS and host CPU, memory and network counters every second
after warm-up, including idle periods between frames. Client durations use a monotonic clock and
include capture, JPEG encoding and the upload/response round trip. Server inference and processing
times are measured independently. Clock differences between computers are not used to infer latency.
Definitions and versions are preserved in run manifests.

A **local functional test**, on the Windows computer rather than Azure, sent recorded-video positions
0, 5 and 10 seconds to the loopback VM service at five-second wall-clock intervals. The service used
one PyTorch inference thread. Its outputs are in `runs/part2-local-server-verified/`, with client
evidence in `runs/part2-local-client-verified/`. The model hash matches Part 1's model.

| Local functional check | Observed result |
|---|---:|
| Successfully received observations | 3 |
| Detected people, in order | 3, 5, 6 |
| Transferred JPEG bytes | 430,137 |
| Client/VM payload hashes | All three match |
| Mean HTTP upload/response duration | 281.333 ms |
| Mean detector-call duration | 208.333 ms |
| Mean capture-to-response duration | 364.667 ms |
| Maximum sampled process RSS in the processing window | 442,093,568 bytes, about 421.6 MiB |

These values establish transport, positive detections, evidence generation and dashboard operation.
They are not cloud performance, capacity or accuracy estimates. Three observations are insufficient
for meaningful tail-latency claims, and host CPU reflects other applications on the computer.
One-second samples can miss brief memory/CPU peaks and exclude startup before monitoring begins.

The analysis script joins client/VM records, checks payload hashes, creates a report graph and prepares
a blank manual-count template. With independent labels, it calculates mean absolute count error and
exact-count agreement, reporting labelled-frame coverage. These count metrics do not establish
bounding-box precision or recall. JPEG payload totals exclude protocol/encryption overhead, while
host network counters include all interfaces and other traffic.

**Still required on Azure:** configuration screenshots, a representative live webcam experiment,
public-IP verification, independent labels, CPU/RAM/bandwidth measurements, meaningful response-time
statistics and Cost Management usage for the VM, disk and public IP. B-series CPU credit state and
experiment duration must be recorded. Student-credit usage is still monetary cost [10]. The report
must interpret these measurements before Part 2 can be considered complete.

## Tasks 3.1-3.6 - Serverless and Azure AI

Not implemented yet. Document the static site, separate raw-image container, input-only Blob Trigger,
Azure AI people detection, output bounding boxes and effectiveness/cost/response-time measurements.

## Tasks 4.1-4.3 - Multiple cameras

Not implemented yet. Define test conditions and capacity criteria, then report increasing real camera
counts at one frame/5 s/camera and comparative performance graphs for both cloud architectures.

## Conclusions

Pending completion of the experiments. Conclusions must refer to observed results and limitations;
no architecture winner or accuracy claim can be established from the current local verification alone.

## References

1. Teaching team. *Practice 1: Cloud Computing Services*. Supplied `Practice1.pdf`, Tasks 1.1-4.3.
2. Teaching team. *Assessment criteria for Practice 1*. Supplied `assessment_criteria_practice_1.pdf`.
3. Stadt Gaggenau. [Webcams](https://www.gaggenau.de/webcams.212.htm). Accessed 12 October 2026.
4. Ultralytics. [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and
   [Predict mode](https://docs.ultralytics.com/modes/predict/). Accessed 12 October 2026.
5. Teaching team. `contador_personas.zip`, original local-processing and webpage example.
6. Teaching team. *Azure documentation 1*, *Azure documentation 2* and
   *Azure Functions with Blob Storage and Event Grid*. Supplied supporting PDFs; cloud work pending.
7. OpenCV. [Pedestrian sample video, vtest.avi, version 4.12.0](https://github.com/opencv/opencv/blob/4.12.0/samples/data/vtest.avi).
   Accessed 12 October 2026.
8. Microsoft. [Bv1 sizes series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/bv1-series).
   Accessed 13 October 2026.
9. Microsoft. [Bsv2 sizes series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/bsv2-series).
   Accessed 13 October 2026.
10. Microsoft. [Plan and manage VM costs](https://learn.microsoft.com/en-us/azure/virtual-machines/cost-optimization-plan-to-manage-costs).
    Accessed 13 October 2026.
