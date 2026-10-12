"""A shared monotonic schedule for local capture and cloud uploads."""

import math


def next_slot(current_slot: float, now: float, interval: float) -> float:
    """Keep the original cadence; skip missed slots instead of creating a burst."""
    return current_slot + max(1, math.ceil((now - current_slot) / interval)) * interval
