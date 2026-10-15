"""Sample CPU, memory and network usage of the VM in the background (Task 2.5)."""

import threading
import time
from datetime import datetime

import psutil

from smart_lighting.results import CsvLog

RESOURCE_FIELDS = (
    "timestamp",
    "cpu_percent",
    "memory_percent",
    "service_memory_mb",
    "network_in_kbps",
    "network_out_kbps",
)
MB = 1024 * 1024


class ResourceMonitor(threading.Thread):
    """Log one resource sample every `interval` seconds; `latest` holds the newest one."""

    def __init__(self, log: CsvLog, interval: float = 5.0):
        super().__init__(daemon=True)
        self.log = log
        self.interval = interval
        self.latest: dict = {}
        self._stopping = threading.Event()

    def run(self) -> None:
        process = psutil.Process()
        psutil.cpu_percent()  # first call primes the counter
        network, started = psutil.net_io_counters(), time.monotonic()
        while not self._stopping.wait(self.interval):
            now, current = time.monotonic(), psutil.net_io_counters()
            seconds = now - started
            self.latest = {
                "timestamp": datetime.now().astimezone(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "service_memory_mb": process.memory_info().rss / MB,
                "network_in_kbps": (current.bytes_recv - network.bytes_recv) * 8 / 1000 / seconds,
                "network_out_kbps": (current.bytes_sent - network.bytes_sent) * 8 / 1000 / seconds,
            }
            self.log.append(**self.latest)
            network, started = current, now

    def stop(self) -> None:
        self._stopping.set()
