"""Sample VM resource usage independently of frame arrival, including idle periods."""

import csv
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil


class ResourceMonitor:
    def __init__(self, output: Path):
        self.output = output
        self.latest: dict = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="resource-monitor", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join()

    def _sample(self) -> None:
        process = psutil.Process()
        cores = psutil.cpu_count() or 1
        process.cpu_percent()
        psutil.cpu_percent()
        previous = psutil.net_io_counters()
        sampled_at = time.monotonic()
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = None
            while not self.stop.wait(1):
                now = time.monotonic()
                memory = psutil.virtual_memory()
                network = psutil.net_io_counters()
                process_cpu = process.cpu_percent()
                row = {
                    "timestamp_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    "sample_seconds": round(now - sampled_at, 6),
                    "host_cpu_percent": psutil.cpu_percent(),
                    "process_cpu_core_percent": process_cpu,
                    "process_cpu_host_percent": round(process_cpu / cores, 3),
                    "process_rss_bytes": process.memory_info().rss,
                    "host_memory_used_bytes": memory.total - memory.available,
                    "host_memory_total_bytes": memory.total,
                    "host_memory_percent": memory.percent,
                    "host_network_sent_bytes": network.bytes_sent - previous.bytes_sent,
                    "host_network_received_bytes": network.bytes_recv - previous.bytes_recv,
                }
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                self.latest = row
                sampled_at, previous = now, network
