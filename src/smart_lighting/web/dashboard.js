// Polls the VM service and redraws the dashboard.
const REFRESH_MS = 5000;
const TIMELINE_POINTS = 120;

const histogram = new Chart(document.getElementById("histogram"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "Frames", data: [], backgroundColor: "#2a7de1" }] },
  options: { scales: { x: { title: { display: true, text: "People in frame" } },
                       y: { title: { display: true, text: "Frames" }, beginAtZero: true } } },
});

const timeline = new Chart(document.getElementById("timeline"), {
  type: "line",
  data: { labels: [], datasets: [{ label: "People", data: [], borderColor: "#e8743b", stepped: true }] },
  options: { scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } },
});

function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/); // handle \r\n
  const fields = header.split(",");
  return lines.map((line) => Object.fromEntries(line.split(",").map((v, i) => [fields[i], v])));
}

const mean = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;
const show = (id, text) => { document.getElementById(id).textContent = text; };

async function refreshCounts() {
  const rows = parseCsv(await (await fetch("counts.csv", { cache: "no-store" })).text());
  if (!rows.length) return;
  const people = rows.map((r) => Number(r.people));

  const frequency = Array(Math.max(...people) + 1).fill(0);
  people.forEach((p) => frequency[p]++);
  histogram.data.labels = frequency.map((_, i) => i);
  histogram.data.datasets[0].data = frequency;
  histogram.update();

  const recent = rows.slice(-TIMELINE_POINTS);
  timeline.data.labels = recent.map((r) => new Date(r.timestamp).toLocaleTimeString());
  timeline.data.datasets[0].data = recent.map((r) => Number(r.people));
  timeline.update();

  show("frames", rows.length);
  show("people", people.at(-1));
  show("mean", mean(people).toFixed(2));
  show("inference", `${mean(rows.map((r) => Number(r.inference_ms))).toFixed(0)} ms`);
  document.getElementById("latest").src = `latest.jpg?t=${Date.now()}`;
}

async function refreshResources() {
  const r = await (await fetch("resources", { cache: "no-store" })).json();
  if (!r.timestamp) return;
  document.getElementById("resources").hidden = false;
  show("cpu", `${r.cpu_percent.toFixed(0)} %`);
  show("memory", `${r.memory_percent.toFixed(0)} %`);
  show("network", `${r.network_in_kbps.toFixed(0)} kbit/s`);
}

async function refresh() {
  await Promise.allSettled([refreshCounts(), refreshResources()]);
}

refresh();
setInterval(refresh, REFRESH_MS);
