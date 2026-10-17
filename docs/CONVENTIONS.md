# Conventions (binding for all steps)

If you must change anything here, update this file in the same change and log it under "Decisions" in
`docs/PROGRESS.md`.

## Tooling

- Python >= 3.12, `src/` layout, package name `hopper`.
- **uv manages everything.** `.mise.toml` pins `uv` (run `mise install` if `uv` is missing). Use only:
  `uv add <pkg>` / `uv add --dev <pkg>` to add dependencies (never edit the dependency lists by hand
  or use `pip`), `uv sync` to install, `uv run <cmd>` to run anything (pytest, ruff, scripts),
  and commit `uv.lock`. Do not create or activate a virtualenv manually.
- Runtime deps: numpy, scipy, pydantic>=2, pyyaml, pandas, pyarrow, matplotlib.
- Dev deps: pytest, hypothesis, ruff.
- Write the EKF by hand (no filterpy). `python-control` and `casadi` arrive in S13.
- Commands: `uv sync`, `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`.

## Units and frames

- SI units only. Angles in radians. Time in seconds.
- **Inertial frame (I): ENU.** x = east, y = north, z = up. Gravity is `[0, 0, -g]`. Ground is z = 0.
- **Body frame (B):** origin at the **centre of mass** (CoM). z_B runs along the vehicle's long axis
  toward the nose (up when upright). x_B, y_B complete a right-handed frame.
  Identity quaternion = vehicle upright with x_B = east, y_B = north.
- **Axial positions** (`z`) on the vehicle are measured along z_B **from the base** (the engine gimbal
  pivot is at z = 0 on the vehicle). The CoM is at `z_com` above the base and moves as propellant burns.
  Position of the gimbal pivot relative to the CoM, in body coordinates, is `[0, 0, -z_com]`.
- Quaternions are **scalar-first** `[w, x, y, z]`, unit norm, and rotate body -> inertial:
  `v_I = R(q) @ v_B`. Renormalise after every integration step. Canonicalise sign (w >= 0) only
  when logging or comparing.
- Angular velocity `omega` is the body-frame rate of B relative to I, expressed in B.
- Euler angles are for display only (ZYX yaw-pitch-roll). Never use them in dynamics or control.

## State vector (flat numpy array, length 19)

Defined once in `hopper/types.py` as index constants.

| Index | Name | Meaning |
|-------|------|---------|
| 0:3 | `POS` | position in I (m) |
| 3:6 | `VEL` | velocity in I (m/s) |
| 6:10 | `QUAT` | attitude quaternion |
| 10:13 | `OMEGA` | angular velocity in B (rad/s) |
| 13 | `PROP` | propellant mass (kg) |
| 14 | `THR` | actual thrust fraction 0..1 (throttle lag state) |
| 15:17 | `GIM` | actual gimbal angles `[gx, gy]` (rad) |
| 17:19 | `GIMR` | gimbal angle rates (rad/s) |

`N_STATE = 19`. Time is passed separately.

## Gimbal and thrust geometry

Gimbal angles `gx` (about x_B) and `gy` (about y_B) rotate the thrust direction from +z_B:

```
d_B = [cos(gx) * sin(gy), -sin(gx), cos(gx) * cos(gy)]    # unit vector
F_B = thrust * d_B
```

Thrust acts at the gimbal pivot. Torque about the CoM: `tau = r x F_B` with `r = [0, 0, -z_com]`.
Expanded: `tau = [z_com * F_y, -z_com * F_x, 0]`. So positive `gy` (thrust toward +x_B) gives a torque
about **-y_B**, and positive `gx` (thrust toward -y_B) gives a torque about **-x_B**. S03 tests pin this.

## Interfaces (in `hopper/types.py`)

```python
@dataclass(frozen=True)
class Command:
    throttle: float        # 0..1 requested thrust fraction (0 = engine off)
    gimbal: np.ndarray     # shape (2,), requested [gx, gy] rad
    roll_torque: float     # -1..1 fraction of max RCS roll torque about z_B

@dataclass(frozen=True)
class SensorData:
    t: float                          # time stamp of the sample set (s)
    gyro: np.ndarray | None           # (3,) rad/s, body frame; None = no new sample
    accel: np.ndarray | None          # (3,) m/s^2 specific force, body frame
    baro_alt: float | None            # m above ground reference
    gps_pos: np.ndarray | None        # (3,) m in I
    gps_vel: np.ndarray | None        # (3,) m/s in I
    lidar_range: float | None         # m along -z_B (None when out of range or no sample)
```

A `None` field means "no new measurement this tick". Sensors hold their rate; the controller is called
at its own rate with whatever arrived since the last call (latest sample per sensor).

```python
class Controller(Protocol):
    def reset(self) -> None: ...
    def step(self, sensors: SensorData, t: float) -> Command: ...
```

Controllers that cheat with true state (perfect-state mode used in S08/S09) receive an optional
`StateEstimate` instead of deriving it, through a separate method:

```python
@dataclass(frozen=True)
class StateEstimate:
    t: float
    pos: np.ndarray; vel: np.ndarray; quat: np.ndarray; omega: np.ndarray
    mass: float                       # estimated total mass (kg)
```

and `StateFeedbackController.step_state(est: StateEstimate, t) -> Command`. The sim engine builds a
`StateEstimate` from true state in perfect-state mode, or from the EKF later.

## Vehicle config (pydantic, `hopper/config.py`, YAML in `configs/`)

Models are frozen and reject unknown keys. `make_state` renormalises the quaternion and `Command` coerces `gimbal` to a (2,) float array. Field groups (names are the contract; exact defaults are placeholders):

- `mass`: `dry_mass`, `dry_inertia` (3 diag, kg m^2, about dry CoM), `dry_com_z`
- `tank`: `prop_mass_initial`, `bottom_z`, `height`, `radius` (uniform-density cylinder)
- `engine`: `thrust_max`, `thrust_min_frac`, `isp`, `throttle_tau`, `gimbal_max`,
  `gimbal_rate_max`, `gimbal_wn`, `gimbal_zeta`
- `rcs`: `max_roll_torque`
- `legs`: `count`, `radius` (foot circle), `length`, `stiffness`, `damping`, `friction`
- `aero`: `cd_axial`, `cd_lateral`, `area_axial`, `area_lateral`, `cp_z`
- `sensors`: filled in S07
- `sim`: `dt` (default 0.001), `controller_rate`, `max_time`

Placeholder vehicle for tests: ~100 kg total (80 kg dry + 20 kg propellant), thrust_max 2500 N,
min throttle 40% (1000 N), gimbal limit 8 deg, 60 deg/s rate, 4 legs on a 0.6 m radius circle.
Note that minimum thrust (1000 N) is roughly equal to full-vehicle weight (~981 N), so hover is only
marginal when full and easy when light. This is typical of real hoppers; keep it in mind for S09.

## Randomness

`numpy.random.Generator` passed explicitly (`rng`). Never use the global numpy RNG. Seeds are
recorded in logs.

## Logging schema (S06)

One row per physics step (or decimated): `t`, the 19 state entries, true total mass, `cmd_*`, thrust
(N), `contact_force`, `event`. Parquet via pandas. Column names are in `hopper/analysis/logger.py`.

## Testing conventions

- `tests/` mirrors `src/hopper/`. Test names describe the physical behaviour.
- Analytic tests use tight tolerances (state them). Stochastic tests use fixed seeds and statistical bounds.
- A step's tests must pass without any other step's optional features.
