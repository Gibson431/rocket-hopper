"""Command delay queue, in physics steps (same rule as the legacy engine)."""

from collections import deque


class CommandQueue:
    """A command computed at step `k` applies from step `k + delay_steps` on; until then the
    previous one is held. Lives inside `ControllerHost` so the delay adds no extra latency
    (docs/DRAKE_DESIGN.md Q9a)."""

    def __init__(self, delay_steps: int):
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self.delay_steps = delay_steps
        self._pending: deque = deque()  # (first step it applies to, command)

    def push(self, step: int, command) -> None:
        self._pending.append((step + self.delay_steps, command))

    def pop_ready(self, step: int, current):
        """The newest command due at `step`, or `current` if none is."""
        while self._pending and self._pending[0][0] <= step:
            current = self._pending.popleft()[1]
        return current

    def clear(self) -> None:
        self._pending.clear()
