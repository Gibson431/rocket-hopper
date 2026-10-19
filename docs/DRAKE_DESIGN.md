# Drake port: design decisions (S12)

Produced by the S12 spike. Spike code is in `scripts/spikes/` (`s12_*.py`, ruff-excluded, throwaway);
the numbers below come from running it on pydrake 1.58, Python 3.14.8, Linux x86_64. Open-loop test case
for "burn" results: 12 s, throttle 0.4-1.0, gimbal pulses, roll torque, 100 kg -> 90 kg, initial spin;
truth = legacy RK4 at dt = 0.25 ms (legacy 1 ms and 0.5 ms agree with it to 4e-10 m).

**Go/no-go: GO.** Nothing found makes the port unattractive. Conditions: the contact model (Q4) is
approximate and S17 must treat force history as unmatched; the port gives parity, not a speedup (Q8).

## Dependency and interpreter

- `uv add drake` resolves `drake 1.58.0` (+ `mosek 11.1.2`, `pydot`) on **Python 3.14.8**, the interpreter uv
  picks for this project. The wheel is `cp312-abi3`, so Python 3.12-3.14 all work.
- Wheels exist for Linux x86_64/aarch64 and macOS arm64 only (no Windows). `mosek` is pulled in only for
  Python < 3.15 on non-aarch64 (lock marker).
- Footprint in the venv: pydrake 146 MB + mosek 46 MB.
- `uv sync`, `uv run pytest` (all existing tests), `uv run ruff check .` and `ruff format --check .` pass with
  the dependency; CI (`uv sync` then the same commands) needs no change.

**Decision:** keep `drake>=1.58.0` as a normal runtime dependency; no interpreter pin.

## Q1. Variable mass

One floating `MultibodyPlant` body follows the burn. After every `AdvanceTo(t + dt)` an outer loop writes
`RigidBody.SetSpatialInertiaInBodyFrame(plant_context, SpatialInertia.MakeFromCentralInertia(m, [0,0,com_z], I_com))`
from the propellant state (full spatial inertia, never `SetMass` + `SetCenterOfMassInBodyFrame`).
Two details decide the accuracy (open-loop burn, RK3 1 ms, error vs truth after 12 s; position / velocity):

| variant | position | velocity |
|---|---|---|
| mass at step start, pivot state kept | 2.0e-2 m | 2.5e-3 m/s |
| + mass predicted at the step midpoint (`prop + 0.5 dt * dprop/dt`) | 1.6e-2 m | 1.9e-3 m/s |
| + CoM velocity kept continuous across the update (velocity only) | 7.7e-3 m | 1.0e-6 m/s |
| + CoM position and velocity kept continuous (**chosen**) | **1.2e-6 m** | **1.0e-6 m/s** |

(attitude 1e-8 rad, omega 1e-8 rad/s, propellant 1e-10 kg for the chosen variant.) The legacy state is the CoM
and is continuous when the CoM moves inside the body; Drake's state is the body origin. So each update
reads the CoM pose/velocity with the old `com_z`, changes the inertia, and rewrites the pivot pose and
velocity so the CoM state is unchanged (`p_o = p_com - R r`, `v_o = v_com - w x r`, ~1e-5 m per step).
Plant gravity must be set to the legacy value: `plant.mutable_gravity_field().set_gravity_vector([0,0,-9.80665])`
(Drake's default 9.81 alone costs 0.02 m/s after 6 s).

Mechanisms compared (6 s constant-thrust burn, 1 ms):

| mechanism | steps/s | note |
|---|---|---|
| (a) outer `AdvanceTo` loop, parameters written between steps | 2150 | supported API; the loop also hosts event checks. **Chosen** |
| (b) periodic publish event mutating a captured root context | 3160 | works, but writes state from a const publish; unspecified ordering with other events; only 1.5x faster |
| (c) custom `LeafSystem` rigid body (legacy ODE, mass as state) | 4700 | exact parity (2e-10 m) but gives up `MultibodyPlant` (geometry, contact, Meshcat, linearisation) |

**Decision:** (a). The loop step equals the physics step (1 ms). A `LeafSystem` cannot write the plant's
parameters (an event can only modify its own state), which is why the writer lives in the sim loop.

## Q2. Body frame

**Decision:** Drake body frame B = legacy body axes with its **origin at the engine gimbal pivot** (the base,
z = 0 on the vehicle). Thrust is a spatial force at the origin with no torque from the offset; the roll
RCS torque is a pure torque; Drake computes `r x F` about the CoM itself. Verified against the spike:

- `QuaternionFloatingJoint` state is `[q (w,x,y,z), p_WBo, w_WB_W, v_WBo_W]` (13). **Angular velocity is
  expressed in the world frame** (legacy: body frame), the translation is the **origin's** velocity.
  `MapVelocityToQDot` equals the legacy `quat_derivative(q, w_B)` to 1e-17 when fed `R w_B`.
- Legacy (CoM) to Drake: `R = R(q)`, `r = R [0,0,com_z(prop)]`, `w_I = R w_B`, `p_o = p - r`, `v_o = v - w_I x r`.
  Drake to legacy: the inverse; `w_B = R^T w_I`. The quaternion needs no change (both scalar-first, body to world).
- `SetFreeBodySpatialVelocity(context, body, V)` takes the context first in 1.58.

S14 round-trip test (legacy to Drake to legacy, random states, 1e-12) belongs to S14.

## Q3. Continuous vs discrete plant

| plant | burn error (pos / att) | speed | drop (diss 2.5) |
|---|---|---|---|
| continuous (`time_step=0`), RK3 1 ms | 1.2e-6 m / 1e-8 rad | 1900 steps/s | peak 19.1 kN, penetration 40 mm |
| discrete 1 ms (SAP) | 0.20 m / 1.5e-2 rad | 2600 | peak 18.1 kN, penetration 39 mm |
| discrete 2 ms | 0.40 m / 3.0e-2 rad | 2900 | peak 18.0 kN, penetration 38 mm |

The discrete plant is first order and not cheaper in this diagram. **Decision:** continuous plant, explicit
fixed-step integrator. The continuous plant with point contact and RK3 1 ms was stable for a resting 3 s,
a 3.1 m/s drop and a 5 m/s landing (no gain of energy, no chatter).

## Q4. Leg contact

Reference (legacy law, 0.5 m drop, 100 kg, 4 legs 30 kN/m + 1.5 kN s/m, mu 0.6): peak force 18.15 kN,
max foot penetration 38.8 mm, rest penetration 8.17 mm, rest force 980.7 N, **no rebound above the ground**.

- **(a) Drake point contact** (sphere r = 10 mm on each foot, half-space ground). Both geometries get stiffness
  `2 k` so the series value is `k` (series of equal stiffnesses halves it: 30 kN/m on both gave 15 kN/m).
  Foot sphere centre sits one radius above the foot point so the contact point stays at the legacy foot.
  Hunt-Crossley `dissipation d` (s/m) sweep, drop from 0.5 m, 1 ms RK3:

  | d | peak force | penetration | rebound |
  |---|---|---|---|
  | 0 | 11.8 kN | 88 mm | bounces back to 0.5 m |
  | 2.0 | 17.6 kN | 44.0 mm | none |
  | **2.5** | **19.1 kN (+5%)** | **40.3 mm (+4%)** | none |
  | 3.0 | 20.5 kN | 37.5 mm | none |

  Rest height and force match exactly (8.17 mm, 980.7 N). Force-time history over the first 0.4 s: rms
  difference 1.5 kN, max 16 kN (the legacy damper steps by c v at impact; Hunt-Crossley ramps with depth).
  Peak force vs legacy by touchdown speed (d = 2.5): 0.7 m/s **-27%** (3.5 vs 4.8 kN), 3.1 m/s +5%, 5 m/s **+25%**
  (36.9 vs 29.6 kN); penetration +21%, +4%, -19%. No rebound in any case. Speed with contact: ~27k steps/s
  (C++ contact; the Python contact observer is what costs time).
  Friction: Drake regularises with `stiction_tolerance`. With mu 0.3 (mu 0.6 tips this 1.5 m-CoM vehicle when
  sliding, so any slide test must use lower mu or compare to legacy, never to `v^2/(2 mu g)`), 1 m/s slide,
  3 s: residual yaw rate from a 1 rad/s spin is 5.8e-3 rad/s at tolerance 1e-4 and 1e-3, 3.6e-5 at 1e-2 and
  4.0e-4 at 5e-2 (all under the 1e-3 target except 1e-4/1e-3; the S05 limit cycle is the 1e-4 case). The
  tolerance 5e-2 matches legacy `V_EPS` and keeps residual slide speed lowest (7.5e-3 m/s). **Open
  risk:** the slide distance is 0.048 m in Drake vs 0.105 m legacy (analytic 0.170 m: legacy itself is short
  because friction pitches the vehicle); S17 must either explain or accept it.
- **(b) prismatic legs + `LinearSpringDamper` + foot sphere** works and is stable at 1 ms RK3 (leg mass 5 kg, foot
  contact 3e5 N/m): 15k steps/s, and a linear damper can equal legacy exactly. But it adds 4 DOF and 4 leg
  masses (20% extra weight unless taken from the dry mass), a second stiffness in series, and a two-sided
  spring. Not tuned further. Rejected for complexity.
- **(c) legacy `contact_loads` as a Python applied force** (applied at the CoM, input = plant state mapped
  through the Q2 adapter): peak force 18151 N, penetration 39.3 mm, i.e. parity to 0.4% (RK3 vs RK4), 1950 steps/s.

**Decision:** (a) is the default contact model with `d = 2.5 s/m`, `stiction_tolerance = 0.05 m/s`, series
stiffness equal to `legs.stiffness`; (c) is kept behind a switch (`contact_model: "legacy"`) for exact-parity tests and
as the fallback if S17 cannot accept the (a) differences. What is matched: rest height and force (exact),
rebound count (0), peak penetration within +-20% and peak force +5% at the 3 m/s drop. What is not: force history,
peak force at hop touchdown speed (-27%). S17 asserts the first group against legacy and physical bounds
(energy never increases, peak below the crash limit) for the second.

## Q5. Integrator

No RK4 exists in pydrake. Compared on the burn (error vs truth after 12 s, position / attitude):

| integrator | dt | position | attitude |
|---|---|---|---|
| RK2 | 1 ms | 9.2e-3 m | 9.4e-5 rad |
| RK3 | 4 ms | 7.6e-5 m | 6.6e-7 |
| RK3 | 2 ms | 9.3e-6 | 8.0e-8 |
| **RK3** | **1 ms** | **1.2e-6** | **9.6e-9** |
| RK3 | 0.5 ms | 4.3e-7 | 1.1e-9 |

Halving dt cuts the error by about 8 (third order). **Decision:** `RungeKutta3Integrator`, fixed step mode,
`max_step_size = 1e-3`, via `ResetIntegratorFromFlags(sim, "runge_kutta3", dt)` then
`set_fixed_step_mode(True)`. Logs stay on the 1 ms grid. No adaptive stepping. Update the project rule in S22
to "fixed-step RK3 at 1 ms; truth for tests is legacy RK4 at 0.25 ms". A Python-side RK4 over
`diagram.CalcTimeDerivatives` is possible but would bypass Drake's event handling; not needed at 1e-6 m.

**Hazard found (applies to every system):** a time-dependent command read inside a derivative or output
callback is stale for one RK stage after the change (error 6.7e-4 in throttle at each switch, 1 step lag in
propellant). Commands must live in **discrete state** updated by a periodic event; then the engine matches
legacy to 1e-10.

## Q6. Accelerometer

`plant.get_body_spatial_accelerations_output_port()` gives `A_WBo_W` (origin acceleration, angular acceleration).
Specific force at the CoM (legacy IMU mount) in body axes:
`f_B = R^T (a_o + alpha x r + w x (w x r) + [0,0,g])` with `r = R [0,0,com_z]`. During the burn it agrees with the
legacy `info["specific_force_B"]` to **1.9e-4 m/s^2** worst over 24 samples (IMU noise is ~2e-2). The port also
evaluates on a discrete plant (not needed). With point contact the port includes contact forces, so impact
spikes are seen by the IMU. **Decision:** use this port; no separate force bookkeeping.

## Q7. Randomness and determinism

**Decision:** every random source is a `numpy.random.Generator` owned by the system that uses it, built in the
system constructor from a seed derived from the run seed (same `SeedSequence` spawn order as legacy:
sensors `rng.spawn(4)` etc.), and drawn **only in periodic discrete-update events** (never in derivative or output
callbacks, which RK stages call several times) and stored in discrete state. Verified with an OU gust drawing
3 normals per ms: two runs with the same seed are bit-identical, a run after an unrelated simulation in the
same process is bit-identical, a different seed differs. A diagram is built fresh per run (never rewound).

## Q8. Throughput

RK3 1 ms, free flight, plant + OU gust + n Python force systems (small numpy) + combiner:

| n Python systems | `AdvanceTo(T)` | outer loop with per-step mass write |
|---|---|---|
| 0 | 23.3k steps/s | 19.1k |
| 1 | 12.7k | 11.3k |
| 3 | 6.8k | 6.2k |
| 6 | 4.1k | 3.8k |

Legacy open loop: 4.0k steps/s (1.3k with controller and sensors, ~1 EKF hop per minute per worker). The
outer-loop mass write costs ~10%. So the physics side with 6 Python systems is level with legacy; **the port is
parity, not a speedup**, and Monte Carlo throughput is still limited by the controller and EKF in Python.
Keep on the hot path only what is needed: one combiner, one propulsion system, one aero/wind system, the contact
observer only on log ticks. Contact itself is in C++. **S18 budget: no slower than 0.7x legacy (about 900 steps/s
for a full EKF hop); achievable (physics alone ~4k).**

## Q9a. Controller timing

Simultaneous discrete updates all read the pre-update context. A separate controller (100 Hz) and 1 kHz delay-0
latch give a **one-step (1 ms) latency**: integral of the applied command over 11 ms is 10.0 units vs legacy 12.0.
With the controller and the delay in **one system** (one 1 kHz update that runs the controller on its ticks and
applies the command in the same update) it is 12.0, exactly legacy. **Decision:** `ControllerHost` owns controller
call, delay queue and applied command in one system; engine, faults and plant read the applied command from its
discrete state.

## Q9. Multiprocessing

`MultibodyPlant` is not picklable (TypeError). Workers receive only the scenario and seed, build the diagram
inside the worker, and run. A `ProcessPoolExecutor(3)` on 6 jobs used 3 PIDs and each result equals the serial
result exactly. (Python 3.14's default `forkserver` start method re-imports the main module: guard scripts with
`if __name__ == "__main__"`.)

## Proposed parity tolerances (S13-S20)

Legacy self-convergence (dt = 1 ms vs 0.5 ms): open loop 4e-10 m. Closed-loop perfect-state hop (22.5 s flight): apogee
2e-8 m, touchdown speed 1e-8 m/s, landing error 5e-5 m, trajectory position max 1.3e-4 m (rms 6.5e-6), velocity
max 9e-3 m/s (at the impact; rms 1.2e-4), omega max 6.9e-3 rad/s (impact), propellant 5e-8 kg.

| test | tolerance (measured Drake vs legacy / legacy self) |
|---|---|
| burn, no contact (S14-S16): position, velocity | 1e-5 m, 1e-5 m/s (measured 1.2e-6, 1.0e-6) |
| burn: attitude, omega, propellant | 1e-6 rad, 1e-6 rad/s, 1e-8 kg (1e-8, 1e-8, 1e-10) |
| specific force | 1e-3 m/s^2 (1.9e-4) |
| hop before first touchdown: position, velocity | 2 mm, 5 mm/s (provisional: confirm in S18, 10x legacy self max) |
| hop summary: apogee, flight time, fuel | 1e-3 m, 0.01 s, 1e-3 kg |
| touchdown speed (pre-contact) | 0.05 m/s; landing error 0.05 m (provisional, contact model differs) |
| drop: rest foot height, rest force | 0.1 mm, 1 N (measured exact) |
| drop: peak force, penetration | not matched on force (see Q4); penetration within 25% at 3 m/s |

## Decisions summary

Plant: one floating body, frame at the gimbal pivot, continuous, RK3 1 ms fixed; gravity 9.80665; mass written
by an outer step loop with midpoint mass and CoM-continuity correction; commands in discrete state; IMU from the
body spatial acceleration port; per-system Generators drawn in discrete events; controller + delay in one system;
point contact (d 2.5 s/m, stiction tolerance 0.05 m/s, series stiffness) with the legacy law as switchable fallback;
diagram built inside each worker.
