import time

import numpy as np
import pandas as pd
import pytest

from hopper.control.interface import OpenLoopController
from hopper.dynamics.environment import Environment, Wind
from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.vehicle import vehicle_derivative
from hopper.sim.engine import SimResult, Simulation
from hopper.sim.events import Outcome
from hopper.types import PROP, Command, make_state

OFF = Command(0.0, np.zeros(2), 0.0)


def with_sim(cfg, **kw):
    return cfg.model_copy(update={"sim": cfg.sim.model_copy(update=kw)})


def space_env():
    """No gravity or drag: a vehicle at rest stays at rest."""
    return Environment(gravity_enabled=False, aero_enabled=False)


def high_state(z=50.0):
    return make_state([0, 0, z], [0, 0, 0], None, [0, 0, 0], 20.0)


def sim_for(cfg, controller, env=None, x0=None, seed=0, **kw):
    return Simulation(
        cfg,
        controller,
        env or space_env(),
        high_state() if x0 is None else x0,
        np.random.default_rng(seed),
        **kw,
    )


def row_state(df, k):
    cols = (
        [f"pos_{a}" for a in "xyz"]
        + [f"vel_{a}" for a in "xyz"]
        + [f"quat_{a}" for a in "wxyz"]
        + [f"omega_{a}" for a in "xyz"]
        + ["prop", "thr", "gim_x", "gim_y", "gimr_x", "gimr_y"]
    )
    return df[cols].iloc[k].to_numpy(dtype=float)


class Counting:
    def __init__(self):
        self.n = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step(self, sensors, t):
        raise AssertionError("perfect-state mode must use step_state")

    def step_state(self, est, t):
        self.n += 1
        return OFF


class Alternating(Counting):
    def step_state(self, est, t):
        self.n += 1
        return Command(0.0, np.array([0.01 * (-1) ** self.n, 0.0]), 0.0)


def test_liftoff_acceleration_matches_vehicle_derivative(cfg):
    foot_height = mass_properties(20.0, cfg).com_z + cfg.legs.length
    x0 = make_state([0, 0, foot_height], [0, 0, 0], None, [0, 0, 0], 20.0)
    cmd = Command(0.5, np.zeros(2), 0.0)
    sim = sim_for(cfg, OpenLoopController(cmd), Environment.default(), x0)
    res = sim.run(t_end=2.0)
    df = res.log
    assert df["pos_z"].iloc[-1] > foot_height + 0.5
    assert df["thrust"].iloc[1500] > 9.80665 * df["mass"].iloc[1500]
    k, dt = 1000, cfg.sim.dt
    a_log = (row_state(df, k + 1)[3:6] - row_state(df, k - 1)[3:6]) / (2 * dt)
    a_ref = vehicle_derivative(df["t"].iloc[k], row_state(df, k), cmd, cfg)[3:6]
    assert a_log == pytest.approx(a_ref, abs=2e-3)
    assert a_ref[2] > 1.0


def test_controller_called_at_its_rate(cfg):
    ctl = Counting()
    sim_for(cfg, ctl).run(t_end=10.0)
    assert ctl.n == 1000
    assert ctl.resets == 1


def test_controller_rate_follows_config(cfg):
    ctl = Counting()
    sim_for(with_sim(cfg, controller_rate=50.0), ctl).run(t_end=2.0)
    assert ctl.n == 100


def test_command_is_held_between_controller_calls(cfg):
    res = sim_for(cfg, Alternating()).run(t_end=0.1)
    gx = res.log["cmd_gim_x"].to_numpy()[:100]
    expected = np.array([0.01 * (-1) ** (k // 10 + 1) for k in range(100)])
    assert gx == pytest.approx(expected, abs=1e-15)


def test_command_delay_shifts_applied_command(cfg):
    on = Command(0.7, np.array([0.02, 0.0]), 0.1)
    ctl = OpenLoopController(schedule=[(0.5, on)])
    res = sim_for(cfg, ctl, command_delay=0.05).run(t_end=1.0)
    df = res.log
    first_on = df["t"][df["cmd_throttle"] > 0].iloc[0]
    assert first_on == pytest.approx(0.55, abs=cfg.sim.dt)
    # Before the first command arrives the engine is off with zero gimbal.
    early = df[df["t"] < 0.5]
    assert (early["cmd_throttle"] == 0).all()
    assert (early["cmd_gim_x"] == 0).all()
    assert df["cmd_gim_x"].iloc[-1] == 0.02


def test_command_without_delay_applies_at_the_call(cfg):
    on = Command(0.7, np.zeros(2), 0.0)
    res = sim_for(cfg, OpenLoopController(schedule=[(0.5, on)])).run(t_end=1.0)
    first_on = res.log["t"][res.log["cmd_throttle"] > 0].iloc[0]
    assert first_on == pytest.approx(0.5, abs=cfg.sim.dt)


def test_command_is_constant_across_rk4_stages(cfg):
    # Holding gimbal command fixed: the actuator state follows the same ODE as integrating
    # vehicle_derivative with that single command (no stage-dependent command).
    cmd = Command(0.6, np.array([0.05, 0.0]), 0.0)
    res = sim_for(cfg, OpenLoopController(cmd)).run(t_end=0.05)
    final = row_state(res.log, len(res.log) - 1)
    from hopper.sim.integrator import integrate

    ref, _ = integrate(
        lambda t, x: vehicle_derivative(t, x, cmd, cfg, space_env()),
        high_state(),
        0.0,
        res.t_end,
        cfg.sim.dt,
    )
    assert final == pytest.approx(ref, abs=1e-12)


def test_same_seed_is_bit_identical_and_different_seed_differs(cfg):
    def run(seed):
        wind = Wind(np.array([2.0, 0, 0]), 3.0, 1.0, np.random.default_rng(99))
        env = Environment(wind=wind)
        cmd = Command(0.0, np.zeros(2), 0.0)
        return sim_for(cfg, OpenLoopController(cmd), env, seed=seed).run(t_end=2.0)

    a, b, c = run(1), run(1), run(2)
    pd.testing.assert_frame_equal(a.log, b.log, check_exact=True)
    assert not np.array_equal(a.log["pos_y"].to_numpy(), c.log["pos_y"].to_numpy())
    assert a.seed == 1 and c.seed == 2


def test_drop_ends_in_crash(cfg):
    foot = mass_properties(20.0, cfg).com_z + cfg.legs.length
    x0 = make_state([0, 0, foot + 10.0], [0, 0, 0], None, [0, 0, 0], 20.0)
    res = sim_for(cfg, OpenLoopController(), Environment.default(), x0).run()
    assert res.outcome is Outcome.CRASHED
    assert res.log["event"].iloc[-1] == "CRASHED"
    assert res.t_end == pytest.approx(res.log["t"].iloc[-1])


def test_short_max_time_gives_timeout(cfg):
    res = sim_for(with_sim(cfg, max_time=0.5), OpenLoopController()).run()
    assert res.outcome is Outcome.TIMEOUT
    assert res.t_end == pytest.approx(0.5, abs=cfg.sim.dt)


def test_t_end_before_max_time_leaves_outcome_running(cfg):
    res = sim_for(cfg, OpenLoopController()).run(t_end=0.3)
    assert res.outcome is Outcome.RUNNING
    assert res.t_end == pytest.approx(0.3, abs=1e-9)


def test_unpowered_drop_reaches_ground_with_kinetic_energy_m_g_h(cfg):
    foot = mass_properties(20.0, cfg).com_z + cfg.legs.length
    z0 = foot + 5.0
    x0 = make_state([0, 0, z0], [0, 0, 0], None, [0, 0, 0], 20.0)
    env = Environment(aero_enabled=False)
    res = sim_for(cfg, OpenLoopController(), env, x0).run()
    assert res.outcome in (Outcome.LANDED, Outcome.CRASHED)
    df = res.log
    k = int(np.argmax(df["n_contacts"].to_numpy() > 0)) - 1  # last airborne row
    v = df["vel_z"].iloc[k]
    ke_per_kg = 0.5 * v**2
    assert ke_per_kg == pytest.approx(9.80665 * (z0 - df["pos_z"].iloc[k]), rel=1e-9)
    # Within one step of touchdown, so within ~v*dt of the full 5 m drop.
    assert z0 - df["pos_z"].iloc[k] == pytest.approx(5.0, abs=0.02)


def test_log_records_mass_and_thrust(cfg):
    res = sim_for(cfg, OpenLoopController(Command(1.0, np.zeros(2), 0.0))).run(t_end=1.0)
    df = res.log
    assert df["mass"].iloc[0] == pytest.approx(100.0)
    assert df["mass"].iloc[-1] < df["mass"].iloc[0]
    assert df["thrust"].iloc[-1] == pytest.approx(2500.0, rel=1e-3)
    assert df["prop"].iloc[-1] == pytest.approx(df["mass"].iloc[-1] - cfg.mass.dry_mass, abs=1e-9)
    assert df["prop"].iloc[0] == 20.0 and PROP == 13


def test_non_perfect_state_mode_not_implemented_yet(cfg):
    sim = sim_for(cfg, OpenLoopController(), perfect_state=False)
    with pytest.raises(NotImplementedError, match="S07"):
        sim.run(t_end=0.1)


def test_result_carries_seed_and_config_hash(cfg):
    res = sim_for(cfg, OpenLoopController(), seed=5).run(t_end=0.05)
    assert isinstance(res, SimResult)
    assert res.seed == 5
    assert res.log.attrs["seed"] == 5
    other = with_sim(cfg, max_time=10.0)
    res2 = sim_for(other, OpenLoopController(), seed=5).run(t_end=0.05)
    assert res.log.attrs["config_hash"] != res2.log.attrs["config_hash"]


def test_ten_second_run_is_fast_enough(cfg):
    sim = sim_for(cfg, OpenLoopController(Command(0.5, np.zeros(2), 0.0)), Environment.default())
    t0 = time.perf_counter()
    res = sim.run(t_end=10.0)
    elapsed = time.perf_counter() - t0
    assert len(res.log) > 9000
    assert elapsed < 10.0
