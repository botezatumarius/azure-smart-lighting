"use strict";

const byId = id => document.getElementById(id);
const namespace = "http://www.w3.org/2000/svg";
let latestId;

function element(parent, tag, attributes, text) {
  const node = document.createElementNS(namespace, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function axes(svg, maximum, label) {
  svg.replaceChildren();
  element(svg, "path", {d: "M45 25V215H460", fill: "none", stroke: "#a0b1a6"});
  for (let tick = 0; tick <= 4; tick++) {
    const y = 215 - tick * 45;
    element(svg, "line", {x1: 45, y1: y, x2: 460, y2: y, stroke: "#e7ece7"});
    element(svg, "text", {x: 36, y: y + 4, "text-anchor": "end", "font-size": 12},
      (maximum * tick / 4).toFixed(maximum % 4 === 0 ? 0 : 1));
  }
  element(svg, "text", {x: 45, y: 15, "font-size": 12, fill: "#5b6c65"}, label);
}

function charts(rows) {
  const timeline = byId("timeline");
  const histogram = byId("histogram");
  const maximum = Math.max(1, ...rows.map(row => row.person_count));
  axes(timeline, maximum, "People");
  if (!rows.length) { axes(histogram, 1, "Frames"); return; }
  const recorded = rows.every((row, index) => row.input_type === "recorded_video_test" &&
    (index === 0 || row.video_time_s > rows[index - 1].video_time_s));
  const times = rows.map(row => recorded ? row.video_time_s : Date.parse(row.captured_utc));
  const duration = Math.max(1, times.at(-1) - times[0]);
  const points = rows.map((row, index) => [
    45 + (times[index] - times[0]) / duration * 415,
    215 - row.person_count / maximum * 180,
  ]);
  element(timeline, "polyline", {points: points.map(point => point.join(",")).join(" "),
    fill: "none", stroke: "#237659", "stroke-width": 2});
  for (const [x, y] of points) element(timeline, "circle", {cx: x, cy: y, r: 2.5, fill: "#237659"});
  const label = row => recorded ? `${row.video_time_s}s` : row.captured_utc.slice(11, 19);
  element(timeline, "text", {x: 45, y: 238, "font-size": 12}, label(rows[0]));
  element(timeline, "text", {x: 460, y: 238, "text-anchor": "end", "font-size": 12}, label(rows.at(-1)));
  byId("timeline-note").textContent = recorded ? "Recorded-video positions in seconds" :
    "Acquisition time in UTC";
  const bins = new Map();
  const binWidth = Math.max(1, Math.ceil((maximum + 1) / 15));
  for (let start = 0; start <= maximum; start += binWidth) bins.set(start, 0);
  for (const row of rows) {
    const start = Math.floor(row.person_count / binWidth) * binWidth;
    bins.set(start, bins.get(start) + 1);
  }
  const frequencies = [...bins];
  const maxFrequency = Math.max(...bins.values());
  axes(histogram, maxFrequency, "Frames");
  const width = 415 / frequencies.length;
  frequencies.forEach(([count, frequency], index) => {
    const binLabel = binWidth === 1 ? count : `${count}–${count + binWidth - 1}`;
    const x = 45 + index * width;
    const height = frequency / maxFrequency * 180;
    const bar = element(histogram, "rect", {x: x + width * .15, y: 215 - height,
      width: width * .7, height, rx: 3, fill: "#237659"});
    element(bar, "title", {}, `${binLabel} people: ${frequency} frames`);
    element(histogram, "text", {x: x + width / 2, y: 238, "text-anchor": "middle", "font-size": 12}, binLabel);
  });
}

async function refresh() {
  try {
    const response = await fetch("/results", {cache: "no-store"});
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    const latest = data.rows.at(-1);
    byId("frames").textContent = data.total_frames;
    byId("people").textContent = latest ? latest.person_count : "—";
    const metrics = data.resources;
    byId("cpu").textContent = metrics.host_cpu_percent === undefined ? "—" : `${metrics.host_cpu_percent.toFixed(1)}%`;
    byId("ram").textContent = metrics.process_rss_bytes === undefined ? "—" : `${(metrics.process_rss_bytes / 1048576).toFixed(0)} MiB`;
    charts(data.rows);
    if (latest) {
      byId("details").textContent = `${latest.camera_id} · ${latest.input_type === "recorded_video_test" ?
        "Recorded video test" : "Live webcam"} · ${latest.width} × ${latest.height} · ` +
        `${latest.inference_ms.toFixed(1)} ms inference · ${(latest.payload_bytes / 1024).toFixed(1)} KiB uploaded`;
      byId("caption").textContent = `Captured ${latest.captured_utc} · Frame ${latest.frame_id}`;
      if (latest.frame_id !== latestId) {
        byId("latest").src = `/latest.jpg?frame=${encodeURIComponent(latest.frame_id)}`;
        byId("latest").hidden = false;
        latestId = latest.frame_id;
      }
    }
    const age = latest ? Math.max(0, (Date.now() - Date.parse(latest.received_utc)) / 1000) : 0;
    byId("status").textContent = latest ? `Service available · latest frame received ${age.toFixed(0)} seconds ago` :
      "Service ready · waiting for frames from the local computer";
    byId("status").className = "";
  } catch (error) {
    byId("status").textContent = `Refresh failed · showing previously received data. ${error.message}`;
    byId("status").className = "error";
  } finally {
    setTimeout(refresh, 5000);
  }
}

refresh();
