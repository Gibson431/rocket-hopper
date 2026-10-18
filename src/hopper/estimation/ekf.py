"""Error-state (multiplicative) extended Kalman filter for position, velocity and attitude.

Nominal state (what `p`, `v`, `q`, `bg`, `ba`, `bb` hold):
    p   position of the CoM in I (m)             v   velocity in I (m/s)
    q   attitude, scalar-first, body -> inertial
    bg  gyro bias (rad/s, body)                  ba  accelerometer bias (m/s^2, body)
    bb  barometer bias (m)

Error state (16), always "truth minus estimate", with the attitude error in the body frame:
    dp (0:3), dv (3:6), dtheta (6:9), dbg (9:12), dba (12:15), dbb (15)
    q_true = q (x) exp(dtheta),  p_true = p + dp, ...
`P` is the covariance of this vector, so NEES = e^T P^-1 e uses the same definition.

Sensors:
    IMU      propagates the state. The engine hands over the newest IMU sample at the controller
             rate, so each sample is treated as one measurement with the per-sample noise
             (`gyro_sigma`, `accel_sigma`), integrated with the trapezoid rule over the interval.
    baro     z = p_z + bb + noise
    lidar    range = p_z / R33(q) (flat ground at z = 0), used while the estimated tilt < 45 deg
    GPS      position and velocity in I
    ZUPT     optional zero-velocity pseudo-measurement when the vehicle is known to be on the pad

Latency: every measurement carries its sample time. The filter keeps a history of states, the raw
IMU inputs and the measurements applied at each step. A late measurement is attached to the history
step nearest its sample time and the history is re-propagated from there, so the update sees the
state it actually describes. Setting a sensor latency to 0 in the params gives the naive filter.
The filter's time axis is the IMU measurement time: a frame stamped with the controller call
time `t` describes the vehicle `imu_delay` earlier (latency plus the age of the newest sample).
A sensor sample taken at `s` is therefore applied at `s + imu_delay`. The estimate is not
extrapolated over that gap.

Mass is not a filter state; see `hopper.estimation.mass`. Yaw has no reference (no magnetometer):
it starts at 0 and is only weakly observable through horizontal acceleration against GPS.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hopper.estimation.mass import PropellantEstimator
from hopper.math3d import quat_from_euler_zyx, quat_mul, quat_normalize, quat_to_dcm, skew
from hopper.types import StateEstimate

N_ERR = 16
_P, _V, _TH, _BG, _BA, _BB = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15), 15
_EYE3 = np.eye(3)
_RESEED = 10.0  # covariance added (in units of R) when a long-rejected sensor is trusted again


@dataclass(frozen=True)
class EkfParams:
    # IMU: 1-sigma of one sample at the nominal rate, bias random walks per sqrt(s)
    gyro_sigma: float
    accel_sigma: float
    gyro_bias_rw: float
    accel_bias_rw: float
    gyro_bias_init: float
    accel_bias_init: float
    imu_delay: (
        float  # s: age of the newest IMU sample at the controller call (latency + ~half a period)
    )
    # barometer
    baro_sigma: float
    baro_bias_rw: float
    baro_latency: float
    # GPS
    gps_pos_sigma: tuple[float, float, float]
    gps_vel_sigma: float
    gps_latency: float
    # lidar: sigma = sigma_abs + sigma_rel * range
    lidar_sigma_abs: float
    lidar_sigma_rel: float
    lidar_latency: float
    # vehicle facts for the propellant dead-reckoning
    dry_mass: float
    prop_mass_initial: float
    thrust_max: float
    thrust_min_frac: float
    isp: float
    throttle_tau: float
    # tuning
    gravity: float = 9.80665
    baro_bias_init: float = 0.1  # m
    align_time: float = 1.0  # s of static data averaged for the start-up attitude and position
    init_pos_sigma: tuple[float, float, float] = (0.5, 0.5, 0.5)  # used by `initialise` directly
    init_vel_sigma: float = 0.05  # m/s: the vehicle starts at rest
    init_att_sigma: tuple[float, float, float] = (0.01, 0.01, 0.02)  # rad
    init_yaw_sigma: float = 0.02  # rad, used by the static alignment
    lidar_max_tilt: float = np.radians(45.0)
    zupt_sigma: float = 0.02  # m/s
    # innovation gates on the normalised innovation squared (chi-square 99.99% points or so)
    gate_baro: float = 16.0
    gate_lidar: float = 16.0
    gate_gps: float = 30.0
    gate_zupt: float = 16.0
    history_length: float = 0.5  # s of states kept for late measurements
    reject_timeout: float = 1.0  # s of continuous rejections after which a sensor is trusted again


@dataclass
class _State:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    bg: np.ndarray
    ba: np.ndarray
    bb: float
    P: np.ndarray

    def copy(self) -> "_State":
        return _State(
            self.p.copy(), self.v.copy(), self.q.copy(), self.bg.copy(), self.ba.copy(), self.bb,
            self.P.copy(),
        )  # fmt: skip


@dataclass
class _Meas:
    kind: str  # "baro" | "lidar" | "gps" | "zupt"
    t: float
    value: object
    new: bool = True  # not yet evaluated: only the first evaluation is logged
    accepted: bool = False
    force: bool = False  # skip the gate (set on the first evaluation, kept for replays)


@dataclass
class _Frame:
    t: float
    raw: tuple[np.ndarray, np.ndarray] | None  # (gyro, accel) that led here; None for the first
    pre: _State  # after propagation to `t`, before this frame's measurements
    meas: list[_Meas] = field(default_factory=list)


def _exp_quat(rv: np.ndarray) -> np.ndarray:
    angle = float(np.sqrt(rv @ rv))
    if angle < 1e-12:
        return np.array([1.0, *(0.5 * rv)])
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) / angle * rv)])


class EKF:
    def __init__(self, params: EkfParams) -> None:
        self.params = params
        self._grav = np.array([0.0, 0.0, -params.gravity])
        self.mass = PropellantEstimator(params)
        self.reset()

    # --- state access -------------------------------------------------------------------

    def reset(self) -> None:
        self._s: _State | None = None
        self._frames: deque[_Frame] = deque()
        self._align_buf: dict[str, list] = {
            "t0": [],
            "accel": [],
            "gps": [],
            "baro": [],
            "lidar": [],
        }
        self._gyro_last = np.zeros(3)
        self._throttle = 0.0
        self.mass.reset()
        self.nis_log: list[tuple[str, float, float, int, bool]] = []
        self.innovations: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self.n_rejected = {"baro": 0, "lidar": 0, "gps": 0, "zupt": 0}
        self.n_stale = 0
        self._reject_since: dict[str, float] = {}  # first rejection of the current streak

    @property
    def initialised(self) -> bool:
        return self._s is not None

    @property
    def p(self) -> np.ndarray:
        return self._s.p.copy()

    @property
    def v(self) -> np.ndarray:
        return self._s.v.copy()

    @property
    def q(self) -> np.ndarray:
        return self._s.q.copy()

    @property
    def bg(self) -> np.ndarray:
        return self._s.bg.copy()

    @property
    def ba(self) -> np.ndarray:
        return self._s.ba.copy()

    @property
    def bb(self) -> float:
        return float(self._s.bb)

    @property
    def P(self) -> np.ndarray:  # noqa: N802 (covariance, matrix symbol)
        return self._s.P

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma of the 16 error states (for logging)."""
        return np.sqrt(np.diag(self._s.P))

    def estimate(self) -> StateEstimate | None:
        if self._s is None:
            return None
        s = self._s
        q = s.q if s.q[0] >= 0 else -s.q
        return StateEstimate(
            self._frames[-1].t, s.p.copy(), s.v.copy(), q.copy(), self._gyro_last - s.bg,
            self.mass.mass,
        )  # fmt: skip

    def note_command(self, throttle: float) -> None:
        """The throttle command now in force, used to dead-reckon the propellant."""
        self._throttle = float(throttle)

    # --- initialisation -----------------------------------------------------------------

    def initialise(
        self,
        t: float,
        pos,
        vel,
        quat,
        bg=(0.0, 0.0, 0.0),
        ba=(0.0, 0.0, 0.0),
        bb: float = 0.0,
        pos_sigma=None,
        att_sigma=None,
        att_cov=None,
        att_ba_cov=None,
    ) -> None:
        """Start the filter. Covariances default to the params; `att_cov` (3x3) and `att_ba_cov`
        (attitude/accel-bias cross-covariance, 3x3) replace `att_sigma` for a static alignment."""
        p = self.params
        pos_sigma = p.init_pos_sigma if pos_sigma is None else pos_sigma
        att_sigma = p.init_att_sigma if att_sigma is None else att_sigma
        diag = np.concatenate(
            [
                pos_sigma,
                [p.init_vel_sigma] * 3,
                att_sigma,
                [p.gyro_bias_init] * 3,
                [p.accel_bias_init] * 3,
                [p.baro_bias_init],
            ]
        )
        s = _State(
            np.array(pos, dtype=np.float64),
            np.array(vel, dtype=np.float64),
            quat_normalize(np.array(quat, dtype=np.float64)),
            np.array(bg, dtype=np.float64),
            np.array(ba, dtype=np.float64),
            float(bb),
            np.diag(diag**2),
        )
        if att_cov is not None:
            s.P[_TH, _TH] = att_cov
        if att_ba_cov is not None:
            s.P[_TH, _BA] = att_ba_cov
            s.P[_BA, _TH] = np.asarray(att_ba_cov).T
        self._s = s
        self._frames = deque([_Frame(t, None, s.copy())])
        self.mass.update(t, 0.0)

    def _align(self, data) -> None:
        """Collect static data; initialise once `align_time` has passed."""
        buf, p = self._align_buf, self.params
        if data.accel is None:
            return
        if not buf["t0"]:
            buf["t0"].append(data.t)
        buf["accel"].append(data.accel)
        if data.gps_pos is not None:
            buf["gps"].append(data.gps_pos)
        if data.baro_alt is not None:
            buf["baro"].append(data.baro_alt)
        if data.lidar_range is not None:
            buf["lidar"].append(data.lidar_range)
        if data.t - buf["t0"][0] < p.align_time:
            return

        accel = np.asarray(buf["accel"])
        n = len(accel)
        up = accel.mean(axis=0)
        up = up / np.linalg.norm(up)
        # With yaw = 0 the third row of R is [-sin(pitch), cos(pitch) sin(roll), cos(pitch)
        # cos(roll)]; a static accelerometer reads R^T [0, 0, g], i.e. exactly that row.
        roll = np.arctan2(up[1], up[2])
        pitch = np.arctan2(-up[0], np.hypot(up[1], up[2]))
        q = quat_from_euler_zyx(0.0, pitch, roll)

        pos, pos_sigma = np.zeros(3), np.array(p.init_pos_sigma)
        if buf["gps"]:
            gps = np.asarray(buf["gps"])
            m = len(gps)
            pos[:] = gps.mean(axis=0)
            pos_sigma = np.array(p.gps_pos_sigma) / np.sqrt(m)
        if buf["lidar"]:
            r = float(np.mean(buf["lidar"]))
            pos[2] = r * up[2]  # R33 of the aligned attitude
            pos_sigma[2] = (p.lidar_sigma_abs + p.lidar_sigma_rel * r) / np.sqrt(len(buf["lidar"]))
        elif buf["baro"]:
            pos[2] = float(np.mean(buf["baro"]))
            pos_sigma[2] = np.hypot(p.baro_sigma / np.sqrt(len(buf["baro"])), p.baro_bias_init)
        # The measured vertical is u + (ba + n)/g, so the tilt error is [u]x (ba + n)/g: it is fully
        # correlated with the accelerometer bias, and that correlation has to be in P0.
        var_n = p.accel_sigma**2 / n
        perp = _EYE3 - np.outer(up, up)
        att_cov = (p.accel_bias_init**2 + var_n) / p.gravity**2 * perp
        att_cov = att_cov + p.init_yaw_sigma**2 * np.outer(up, up)
        att_ba_cov = p.accel_bias_init**2 * skew(up) / p.gravity
        self.initialise(
            data.t, pos, np.zeros(3), q, pos_sigma=pos_sigma, att_cov=att_cov, att_ba_cov=att_ba_cov
        )

    # --- propagation --------------------------------------------------------------------

    def _propagate(self, s: _State, raw0, raw1, dt: float) -> None:
        p = self.params
        g1, a1 = raw1
        g0, a0 = (g1, a1) if raw0 is None else raw0
        w = 0.5 * (g0 + g1) - s.bg
        f = 0.5 * (a0 + a1) - s.ba
        q_mid = quat_mul(s.q, _exp_quat(0.5 * dt * w))
        rot = quat_to_dcm(q_mid)
        acc = rot @ f + self._grav
        s.p = s.p + s.v * dt + 0.5 * acc * dt * dt
        s.v = s.v + acc * dt
        s.q = quat_normalize(quat_mul(s.q, _exp_quat(dt * w)))

        fm = np.zeros((N_ERR, N_ERR))
        fm[_P, _V] = _EYE3
        fm[_V, _TH] = -rot @ skew(f)
        fm[_V, _BA] = -rot
        fm[_TH, _TH] = -skew(w)
        fm[_TH, _BG] = -_EYE3
        phi = np.eye(N_ERR) + fm * dt + 0.5 * dt * dt * (fm @ fm)
        q_diag = np.zeros(N_ERR)
        q_diag[_V] = (p.accel_sigma * dt) ** 2
        q_diag[_TH] = (p.gyro_sigma * dt) ** 2
        q_diag[_BG] = p.gyro_bias_rw**2 * dt
        q_diag[_BA] = p.accel_bias_rw**2 * dt
        q_diag[_BB] = p.baro_bias_rw**2 * dt
        pm = phi @ s.P @ phi.T
        pm[np.diag_indices(N_ERR)] += q_diag
        s.P = 0.5 * (pm + pm.T)

    def update_imu(self, t: float, gyro, accel) -> None:
        """Propagate to `t` with one IMU sample (the first sample after `initialise` is held)."""
        gyro, accel = np.asarray(gyro, dtype=np.float64), np.asarray(accel, dtype=np.float64)
        prev = self._frames[-1]
        dt = t - prev.t
        if dt <= 0.0:
            return
        s = self._s.copy()
        self._propagate(s, prev.raw, (gyro, accel), dt)
        self._frames.append(_Frame(t, (gyro, accel), s.copy()))
        self._s = s
        self._gyro_last = gyro
        while len(self._frames) > 2 and self._frames[1].t < t - self.params.history_length:
            self._frames.popleft()

    # --- measurement updates ------------------------------------------------------------

    def _inject(self, s: _State, dx: np.ndarray) -> None:
        s.p = s.p + dx[_P]
        s.v = s.v + dx[_V]
        s.q = quat_normalize(quat_mul(s.q, _exp_quat(dx[_TH])))
        s.bg = s.bg + dx[_BG]
        s.ba = s.ba + dx[_BA]
        s.bb = s.bb + dx[_BB]

    def _kalman(self, s: _State, y, h, r_diag, gate: float):
        """Gated Joseph-form update. Returns `(accepted, nis, S)`; `s` changes only if accepted."""
        pht = s.P @ h.T
        S = h @ pht + np.diag(r_diag)  # noqa: N806
        nis = float(y @ np.linalg.solve(S, y))
        if nis > gate:
            return False, nis, S
        k = np.linalg.solve(S, pht.T).T
        dx = k @ y
        ikh = np.eye(N_ERR) - k @ h
        pn = ikh @ s.P @ ikh.T + (k * r_diag) @ k.T
        self._inject(s, dx)
        # Re-express the covariance in the corrected attitude frame.
        g = np.eye(N_ERR)
        g[_TH, _TH] -= skew(0.5 * dx[_TH])
        pn = g @ pn @ g.T
        s.P = 0.5 * (pn + pn.T)
        return True, nis, S

    def _apply(self, s: _State, m: _Meas, dt_m: float) -> None:
        """Evaluate one measurement against `s`, the state at its frame.

        `dt_m` is the sample time minus the frame time (under half a frame); positions are moved
        ahead by `v * dt_m` in the measurement model so the frame grid does not add timing error.
        """
        p = self.params
        h = np.zeros((1, N_ERR))
        if m.kind == "baro":
            y = np.array([m.value - (s.p[2] + dt_m * s.v[2] + s.bb)])
            h[0, 2], h[0, 5], h[0, _BB] = 1.0, dt_m, 1.0
            r, gate = np.array([p.baro_sigma**2]), p.gate_baro
        elif m.kind == "lidar":
            rot = quat_to_dcm(s.q)
            r33 = rot[2, 2]
            if (
                np.arccos(np.clip(r33, -1.0, 1.0)) > p.lidar_max_tilt
                or s.p[2] + dt_m * s.v[2] <= 0.0
            ):
                self._finish(m, None, False)
                return
            z = s.p[2] + dt_m * s.v[2]
            y = np.array([m.value - z / r33])
            h[0, 2], h[0, 5] = 1.0 / r33, dt_m / r33
            h[0, 6] = z * rot[2, 1] / r33**2  # d(range) / d(roll error)
            h[0, 7] = -z * rot[2, 0] / r33**2
            r = np.array([(p.lidar_sigma_abs + p.lidar_sigma_rel * m.value) ** 2])
            gate = p.gate_lidar
        elif m.kind == "gps":
            pos, vel = m.value
            y = np.concatenate([pos - s.p - dt_m * s.v, vel - s.v])
            h = np.zeros((6, N_ERR))
            h[:3, _P] = _EYE3
            h[:3, _V] = dt_m * _EYE3
            h[3:, _V] = _EYE3
            r = np.concatenate([np.square(p.gps_pos_sigma), [p.gps_vel_sigma**2] * 3])
            gate = p.gate_gps
        else:  # zupt
            y = -s.v.copy()
            h = np.zeros((3, N_ERR))
            h[:, _V] = _EYE3
            r, gate = np.full(3, p.zupt_sigma**2), p.gate_zupt
        if m.new:
            since = self._reject_since.get(m.kind)
            m.force = since is not None and m.t - since > p.reject_timeout
        if m.force:
            # Long disagreement means the filter, not the sensor, is wrong: re-seed the observed
            # states from the measurement by letting it dominate this one update.
            s.P = s.P + h.T @ np.diag(_RESEED * r) @ h
        ok, nis, S = self._kalman(s, y, h, r, np.inf if m.force else gate)  # noqa: N806
        self._finish(m, (y, np.diag(S), nis), ok)

    def _finish(self, m: _Meas, innov, ok: bool) -> None:
        m.accepted = ok
        if not m.new:
            return
        m.new = False
        if ok:
            self._reject_since.pop(m.kind, None)
        else:
            self._reject_since.setdefault(m.kind, m.t)
        if innov is not None:
            y, s_diag, nis = innov
            self.innovations[m.kind] = (y, s_diag, nis)
            self.nis_log.append((m.kind, m.t, nis, len(y), ok))
        if not ok:
            self.n_rejected[m.kind] += 1

    def _frame_index(self, t: float) -> int | None:
        frames = self._frames
        if len(frames) > 1:
            half = 0.5 * (frames[1].t - frames[0].t)
        else:
            half = 0.0
        if t < frames[0].t - half - 1e-9:
            return None
        return min(range(len(frames)), key=lambda i: abs(frames[i].t - t))

    def _insert(self, measurements: list[_Meas]) -> None:
        earliest = None
        for m in measurements:
            k = self._frame_index(m.t)
            if k is None:
                self.n_stale += 1
                m.new = False
                continue
            self._frames[k].meas.append(m)
            earliest = k if earliest is None else min(earliest, k)
        if earliest is not None:
            self._replay(earliest)

    def _replay(self, k: int) -> None:
        frames = self._frames
        s = frames[k].pre.copy()
        for j in range(k, len(frames)):
            fr = frames[j]
            if j > k:
                prev = frames[j - 1]
                self._propagate(s, prev.raw, fr.raw, fr.t - prev.t)
                fr.pre = s.copy()
            for m in fr.meas:
                self._apply(s, m, m.t - fr.t)
        self._s = s

    def _single(self, kind: str, t: float, value) -> bool:
        m = _Meas(kind, t, value)
        self._insert([m])
        return m.accepted

    def update_baro(self, t: float, alt: float) -> bool:
        return self._single("baro", t, float(alt))

    def update_lidar(self, t: float, range_m: float) -> bool:
        return self._single("lidar", t, float(range_m))

    def update_gps(self, t: float, pos, vel) -> bool:
        return self._single(
            "gps", t, (np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64))
        )

    def update_zero_velocity(self, t: float) -> bool:
        return self._single("zupt", t, None)

    # --- sensor frame -------------------------------------------------------------------

    def _sample_time(self, t: float, latency: float) -> float:
        """Filter time of a sample delivered at `t`: taken at `t - latency`, plus the IMU delay."""
        return min(t - latency + self.params.imu_delay, t)

    def process(self, data, grounded: bool = False) -> StateEstimate | None:
        """Apply everything in one `SensorData`. None until the static alignment is done.

        `grounded` adds a zero-velocity update (vehicle known to be resting on the pad).
        """
        p = self.params
        if self._s is None:
            self._align(data)  # the aligned state already contains this sample
            return self.estimate()
        self.mass.update(data.t, self._throttle)
        if data.gyro is not None and data.accel is not None:
            self.update_imu(data.t, data.gyro, data.accel)
        meas = []
        if data.baro_alt is not None:
            meas.append(
                _Meas("baro", self._sample_time(data.t, p.baro_latency), float(data.baro_alt))
            )
        if data.lidar_range is not None:
            meas.append(
                _Meas("lidar", self._sample_time(data.t, p.lidar_latency), float(data.lidar_range))
            )
        if data.gps_pos is not None and data.gps_vel is not None:
            meas.append(
                _Meas("gps", self._sample_time(data.t, p.gps_latency), (data.gps_pos, data.gps_vel))
            )
        if grounded:
            meas.append(_Meas("zupt", data.t, None))
        if meas:
            self._insert(meas)
        return self.estimate()
