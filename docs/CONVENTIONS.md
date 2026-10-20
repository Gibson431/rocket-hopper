# Conventions (binding for all steps)

If you must change anything here, update this file in the same change and log it under "Decisions" in
`docs/PROGRESS.md`.

## Tooling

- Python >= 3.12, `src/` layout, package name `hopper`.
- **uv manages everything.** `.mise.toml` pins `uv` (run `mise install` if `uv` is missing). Use only:
  `uv add <pkg>` / `uv add --dev <pkg>` to add dependencies (never edit the dependency lists by hand
  or use `pip`), `uv sync` to install, `uv run <cmd>` to run anything (pytest, ruff, scripts),
  and commit `uv.lock`. Do not create or activate a virtualenv manually.
- Runtime deps: numpy, scipy, pydantic>=2, pyyaml, pandas, pyarrow, matplotlib, drake (`pydrake`, simulation side only).
- Dev deps: pytest, hypothesis, ruff.
- Write the EKF by hand (no filterpy). Drake is the only simulation backend (see `docs/PLAN.md`, "Drake port"). Flight software (`hopper.control`, `hopper.estimation`) and `hopper.types` / `hopper.math3d` import only the standard library, numpy, scipy and each other; `tests/test_import_boundary.py` enforces the allowlist.
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
- `sensors`: `imu`, `baro`, `gps`, `lidar` groups (rate, latency, noise, range; SI units, rad). `SensorData.t` is the controller call time; samples carry sensor latency; `baro_alt` and `lidar_range` are relative to the CoM height. `vehicle_derivative(..., info=d)` fills `d["contact"]` (`ContactOutput`) and `d["specific_force_B"]`.
- `sim`: `dt` (default 0.001), `controller_rate`, `max_time`, and outcome thresholds (`v_crash`, `crash_tilt`, `tipover_tilt`, `land_*`, `max_range`, `max_alt`; see `hopper.sim.events`). `Command` rejects non-finite fields; throttle/roll ranges are clamped in `propulsion`

Placeholder vehicle for tests: ~100 kg total (80 kg dry + 20 kg propellant), thrust_max 2500 N,
min throttle 30% (750 N), gimbal limit 8 deg, 60 deg/s rate, 4 legs on a 0.6 m radius circle.
Minimum throttle is 30% (750 N), below the dry weight (785 N), so hover is feasible at every mass
(40% was not: 1000 N exceeds even the full 981 N weight).

## Control package (`hopper.control`)

- `ControlParams` (vehicle facts the controller may know) is built from the config by
  `hopper.scenario_glue.control_params_from_config`; the control package never imports the config.
- Altitude in the controller is the foot-tip height above ground (`ControlParams.foot_height`).
- `AttitudeController` (S08) -> `PositionController` (altitude + horizontal, driven by a `Setpoint`) ->
  `MissionController` (state machine + `PositionController`). All are perfect-state
  (`step_state`) until S10 adds the estimator.

## Estimation package (`hopper.estimation`)

- Flight software like `hopper.control`: it may import only `hopper.types` and `hopper.math3d`
  (enforced by `tests/estimation/test_estimation_boundary.py`).
- `EKF(EkfParams)` is an error-state filter (16 states: dp, dv, dtheta, dbg, dba, dbaro; definitions in the
  `ekf.py` docstring). `EkfParams` is built from the sensor config by
  `hopper.scenario_glue.ekf_params_from_config`. `EKF.process(SensorData, grounded=False)` returns a
  `StateEstimate` (None until the 1 s static alignment is done). Mass comes from
  `PropellantEstimator` (dead-reckoned from `EKF.note_command`).
- `hopper.estimated_controller.EstimatedController(ekf_params, mission)` is the `Controller` that runs
  sensors -> EKF -> `MissionController.step_state`; the engine stays off until the filter is aligned.
- Optional controller log hook: a controller with `log_columns` (tuple of names) and `log_values()`
  gets those columns appended to the sim log (before `event`); `EstimatedController` logs `est_*` and
  `sig_*` (16 one-sigma values, error-state order).

## Scenarios, faults, Monte Carlo (S11)

- `hopper.scenario`: strict pydantic `Scenario` (YAML in `configs/scenarios/`), `load_scenario`,
  `build_simulation(scenario, seed, dispersions) -> DrakeSimulation`, `sample_dispersions(scenario, seed)` (a pure function
  of the seed), `run_scenario(scenario, seed, dispersed)`. Controller and EKF are built from the
  *nominal* vehicle config; dispersions and faults change only the simulated truth.
- `hopper.sim.faults`: `Fault` hooks `apply_command`, `modify_params` (+ `param_epoch`),
  `modify_sensors`; `DrakeSimulation(..., faults=())` applies them, and with no faults the diagram has
  no fault systems. Faults act on what the actuators/sensors see, never on the controller's inputs
  directly. `GyroBias` (constant offset) is internal to the `gyro_bias` dispersion, not a YAML fault.
- `hopper.analysis.monte_carlo.run_monte_carlo(path, n, base_seed, workers, out_dir)`: run seeds come
  from `run_seeds(base_seed, n)`; a seed fixes dispersions, noise and gusts, so
  `scripts/run_scenario.py <scenario> --seed S --dispersed` reproduces run S exactly.

## Package layout (after S22)

- `hopper.drake`: the simulation (plant, engine, wind, aero, legs, events, sensors, controller host,
  logger, `DrakeSimulation`, visualisation).
- `hopper.dynamics`: `mass_model` (CoM and inertia vs propellant, used by the simulator, the sensors
  and the controller glue) and `environment` (`G0`, ISA `atmosphere`, `Wind` and `Environment`, which
  only *describe* the environment; `drake.wind.WindSource` draws the gusts).
- `hopper.sim`: `events` (`Outcome`, `ContactOutput`, `check_events`, `EventMonitor`), `faults`,
  `result` (`SimResult`, `config_hash`, `seed_of`).
- `hopper.sensors`, `hopper.analysis` (logger, plots, Monte Carlo), `hopper.scenario`,
  `hopper.scenario_glue` (`run_hop`, `summarize_hop`, parameter bridges), `hopper.estimated_controller`.
- **Integrator rule:** fixed-step only, never adaptive: Drake RK3 at `cfg.sim.dt` (1 ms) via
  `sim.make_simulator`. Mass is written between steps from the midpoint propellant mass.

## Drake package (`hopper.drake`, S14+)

- Simulation side only; `hopper.control`, `hopper.estimation`, `hopper.types`, `hopper.math3d` never import it
  or `pydrake` (tests enforce this). Decisions are in `docs/DRAKE_DESIGN.md`.
- `plant.build_vehicle_plant(cfg, builder=None, finalize=True) -> (plant, scene_graph)`: one floating body
  `"hopper"`, continuous, gravity 9.80665. `sim.make_simulator(diagram, dt=1e-3)`: fixed-step RK3.
- **Body-frame rule:** the Drake body origin is the gimbal pivot (base), not the CoM. The 19-element state vector of `hopper.types`
  holds the CoM with the rate in the body frame; Drake holds origin position/velocity and a world-frame rate.
  `state.to_state_vector(plant, ctx)` / `set_state_vector(plant, ctx, x, cfg)` convert (CoM offset read from
  the spatial inertia in the context; quaternion returned with w >= 0; PROP/THR/GIM/GIMR are 0, owned by other systems).
- `mass.MassUpdater(plant, cfg).update(ctx, prop)` writes the full spatial inertia between `AdvanceTo` steps
  and keeps the CoM position/velocity continuous. The caller passes the step-midpoint propellant mass.
- `propulsion.Engine(cfg, plant)` (S15): input ports `command` ([throttle, gx, gy, roll_torque]) and `plant_state`; outputs `spatial_forces` (thrust + roll torque at the pivot), `propellant` (kg), `thrust` (N). Continuous state = state-vector slots 13..18. Details in PROGRESS.md Decisions.
- `wind.WindSource(mean, sigma, tau, seed, dt)` (S16): output `wind` (3); gust is discrete state, a function of time only. `aero.AeroLoads(cfg, plant, aero_enabled)`: inputs `plant_state`, `wind`, `propellant`; output `spatial_forces`. `combiner.ForceCombiner(n)`: concatenates n force lists for the plant's applied-force port. `build_vehicle_plant(..., gravity_enabled=False)` switches gravity off.

## Randomness

`numpy.random.Generator` passed explicitly (`rng`). Never use the global numpy RNG. Seeds are
recorded in logs.

## Logging schema (S06)

One row per physics step (or decimated): `t`, the 19 state entries, true total mass, `cmd_*`, thrust
(N), `contact_force`, `event`. Parquet via pandas. Column names are in `hopper/analysis/logger.py`. `event` is `""` except on the terminal row (outcome name); seed and config hash are in `df.attrs` and the parquet metadata.

## Testing conventions

- `tests/` mirrors `src/hopper/`. Test names describe the physical behaviour.
- Analytic tests use tight tolerances (state them). Stochastic tests use fixed seeds and statistical bounds.
- A step's tests must pass without any other step's optional features.

## Drake legs and events (S17)

- `hopper.drake.legs.add_legs(plant, scene_graph, cfg) -> LegGeometry` adds ground and foot spheres
  before `Finalize` (parameter mapping in the module docstring). `ContactObserver(plant, legs)` has
  inputs `plant_state`, `contact_results` and outputs `foot_forces` (n), `contact_count`,
  `total_force` (3, world), `base_strike`.
- `hopper.drake.events.DrakeEventMonitor(plant, observer, cfg)`: `evaluate(root_context)` runs
  `check_events`; `as_simulator_monitor(diagram)` is for `Simulator.set_monitor`; the outcome
  is in `.outcome`.

## Drake closed loop (S18)

- `hopper.drake.diagram.build_diagram(cfg, controller, env, rng, command_delay, perfect_state) -> DrakeRig`
  (diagram, plant, engine, host, wind, aero, observer, monitor, mass_updater). Wiring (port names):
  `ControllerHost.command` -> `Engine.command`; `plant.state` -> `Engine.plant_state`, `ControllerHost.plant_state`,
  `AeroLoads.plant_state`, `ContactObserver.plant_state`; `Engine.propellant` -> `AeroLoads.propellant`,
  `ControllerHost.propellant`; `WindSource.wind` -> `AeroLoads.wind`; `Engine.spatial_forces` and
  `AeroLoads.spatial_forces` -> `ForceCombiner` -> plant applied forces; plant contact results -> `ContactObserver`.
  The wind seed is `int(rng.integers(2**32))`, the first draw from `rng`.
- `ControllerHost(cfg, controller, command_delay, perfect_state)` (inputs `plant_state`, `propellant`; output `command`, 4):
  one periodic update at the physics step runs the controller every `controller_rate` ticks, pushes the command
  into `delay.CommandQueue` (delay in physics steps, `round(delay / dt)`) and stores the one due as discrete state.
  Delay 0 acts on the step the command was computed. It forwards the controller's `log_columns` / `log_values()`.
- `DrakeSimulation(cfg, controller, env, x0, rng, perfect_state, command_delay, log_decimation, faults)` runs one
  closed loop: `run(t_end) -> SimResult` (log and `df.attrs` as in the logging schema). Per step: event check on x_k (a terminal row gets no controller
  call), `AdvanceTo(t_k)` runs the host update, log row k, midpoint mass write, `AdvanceTo(t_k + dt)`.
  `run` makes a fresh simulator each call. `scenario_glue.run_hop` and `scenario.run_scenario` build it; there is no other backend.

## Drake sensors and faults (S19)

- `hopper.drake.sensors.SensorsSystem(cfg, plant, rng, faults)`: inputs `plant_state`, `propellant`, `accel` (plant
  body spatial accelerations); output `sensors` (a `SensorData`). One periodic *unrestricted* update per physics step
  runs the `SensorSuite` (generators spawned from `rng` after the wind seed draw) and the
  faults' `modify_sensors`. Drake runs unrestricted updates before discrete ones, so the host sees the samples of the
  same step. `specific_force(context)` is the IMU input (`R^T (a_com - g)`).
- `ControllerHost(..., faults=())` has a `sensors` input unless `perfect_state`; it keeps the newest sample per field
  between controller calls and applies the faults' `apply_command` to the held raw command on every step.
- Parameter faults: the engine and RCS parameters are a numeric parameter of `Engine` (`set_config(context, cfg)`);
  `DrakeSimulation.run` writes the effective config when a fault's `param_epoch` changes. With no faults and
  `perfect_state=True` the diagram has no sensor or fault systems. `build_diagram(..., faults=())`, `DrakeRig.sensors`.

## Visualisation (S21)

- `build_vehicle_plant(..., visual=True)` adds illustration-only geometry (no collision role, no dynamics effect);
  `build_diagram(..., visual=, meshcat=)` and `DrakeSimulation(..., meshcat=)` add a live `MeshcatVisualizer`
  (off by default; never in tests or Monte Carlo runs).
- `hopper.drake.viz.replay(log, cfg, speed=1.0, meshcat=None, fps=60)` plays a log; `record_replay(log, cfg, path)`
  writes a static HTML. The logged position is the CoM; the body frame is at the gimbal pivot.
- `hopper.analysis.plots` functions take a log frame and return `Figure`s; `save_all(log, out_dir)` writes PNGs.
  CLI: `scripts/plot_run.py LOG [--replay] [--html F] [--out DIR]`.
- The replay's time/phase/outcome overlay is a Meshcat button label: visible in live replay, absent from the static HTML.
