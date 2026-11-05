"""Exploration schedules."""
from __future__ import annotations


class LinearSchedule:
    """epsilon(t) goes linearly from `start` to `end` over `duration` env steps, then stays at `end`.

    Example: start=1.0, end=0.05, duration=25_000
      t=0      -> 1.0   (fully random)
      t=12_500 -> 0.525
      t>=25_000 -> 0.05
    A SHORT duration = fast decay: exploits early (good early returns, risk of
    locking into a poor policy). A LONG duration = explores longer (slow start,
    usually better/stabler late performance).
    """

    def __init__(self, start: float, end: float, duration: int):
        self.start, self.end, self.duration = float(start), float(end), max(int(duration), 1)

    def __call__(self, t: int) -> float:
        if t >= self.duration:
            return self.end
        return self.start + (t / self.duration) * (self.end - self.start)
