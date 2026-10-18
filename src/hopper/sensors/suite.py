"""All sensors with per-sensor rate scheduling, latency queues and independent RNG streams."""

from collections import deque

import numpy as np

from hopper.config import VehicleConfig
from hopper.sensors.baro import Baro
from hopper.sensors.gps import Gps
from hopper.sensors.imu import Imu
from hopper.sensors.lidar import Lidar
from hopper.types import SensorData

_EPS = 1e-9
_NAMES = ("imu", "baro", "gps", "lidar")


class _Channel:
    """Fixed-rate schedule plus a delay queue of `(deliver_t, sample_t, value)`."""

    def __init__(self, rate: float, latency: float) -> None:
        self.period = 1.0 / rate
        self.latency = latency
        self.next_t = 0.0
        self.queue: deque[tuple[float, float, object]] = deque()

    def due(self, t: float) -> bool:
        return t >= self.next_t - _EPS

    def advance(self) -> None:
        self.next_t += self.period

    def push(self, t: float, value: object) -> None:
        self.queue.append((t + self.latency, t, value))

    def pop_latest(self, t: float):
        """Newest sample delivered by `t`, or None (older ones delivered together are dropped)."""
        out = None
        while self.queue and self.queue[0][0] <= t + _EPS:
            out = self.queue.popleft()[2]
        return out


class SensorSuite:
    """`sample(t, x, info)` must be called every physics step, with increasing `t`.

    `info` needs `specific_force_B` (see `vehicle_derivative`) on steps where `needs_info(t)`.
    Returns a `SensorData` stamped with the delivery time `t`; a field is None unless a sample
    arrived in this call. Sensors draw from child generators spawned in a fixed order, so
    reconfiguring one sensor leaves the noise of the others unchanged.
    """

    def __init__(self, cfg: VehicleConfig, rng: np.random.Generator) -> None:
        s = cfg.sensors
        r = dict(zip(_NAMES, rng.spawn(len(_NAMES)), strict=True))
        self._imu, self._baro = Imu(s.imu, r["imu"]), Baro(s.baro, r["baro"])
        self._gps, self._lidar = Gps(s.gps, r["gps"]), Lidar(s.lidar, r["lidar"])
        self._ch = {n: _Channel(getattr(s, n).rate, getattr(s, n).latency) for n in _NAMES}

    def needs_info(self, t: float) -> bool:
        return self._ch["imu"].due(t)

    def sample(self, t: float, x: np.ndarray, info: dict | None = None) -> SensorData:
        ch = self._ch
        if ch["imu"].due(t):
            ch["imu"].push(t, self._imu.measure(x, info["specific_force_B"]))
            ch["imu"].advance()
        if ch["baro"].due(t):
            ch["baro"].push(t, self._baro.measure(x))
            ch["baro"].advance()
        if ch["gps"].due(t):
            fix = self._gps.measure(t, x)
            if fix is not None:
                ch["gps"].push(t, fix)
            ch["gps"].advance()
        if ch["lidar"].due(t):
            rng_m = self._lidar.measure(x)
            if rng_m is not None:
                ch["lidar"].push(t, rng_m)
            ch["lidar"].advance()

        imu, fix = ch["imu"].pop_latest(t), ch["gps"].pop_latest(t)
        return SensorData(
            t=t,
            gyro=None if imu is None else imu[0],
            accel=None if imu is None else imu[1],
            baro_alt=ch["baro"].pop_latest(t),
            gps_pos=None if fix is None else fix[0],
            gps_vel=None if fix is None else fix[1],
            lidar_range=ch["lidar"].pop_latest(t),
        )
