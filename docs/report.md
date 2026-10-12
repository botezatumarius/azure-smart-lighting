# Smart Lighting System: Cloud Computing Services

**Working report - not yet a complete submission.**

Course: Architectures and Infrastructures in AI

Authors: To be completed by the student/group

Date: 12 October 2026

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

## Tasks 2.1-2.5 - Azure virtual machine

Not implemented yet. Record configuration, image transport, VM detection, public results webpage,
and measured effectiveness, RAM, CPU, bandwidth, cost and response time when this part is completed.

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
