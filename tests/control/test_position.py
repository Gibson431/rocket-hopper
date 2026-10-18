import numpy as np
import pytest

from hopper.control.attitude import AttitudeGains
from hopper.control.interface import StateFeedbackController
from hopper.control.position import (
    AltitudeController,
    AltitudeGains,
    HorizontalController,
    HorizontalGains,
    PositionController,
    Setpoint,
)
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.math3d import quat_rotate
from hopper.scenario_glue import control_params_from_config
from hopper.sim.engine import Simulation
from hopper.types import Command, StateEstimate, make_state

G = 9.80665
PROP_MASS = 10.0
UP = np.array([0.0, 0.0, 1.0])


@pytest.fixture
def params(cfg):
    return control_params_from_config(cfg)


@pytest.fixture
def alt_ctrl(params):
    return AltitudeController(params, AltitudeGains())


@pytest.fixture
def hor_ctrl():
    return HorizontalController(HorizontalGains())


# --- altitude: unit -----------------------------------------------------------------------


def test_hover_feedforward_is_weight_over_max_thrust(alt_ctrl, params):
    mass = 90.0
    thr = alt_ctrl.update(10.0, 0.0, 10.0, 0.0, mass, 0.0, 0.01).throttle
    assert thr == pytest.approx(mass * G / params.thrust_max, rel=1e-9)


def test_feedforward_grows_with_tilt(alt_ctrl):
    flat = alt_ctrl.update(10.0, 0.0, 10.0, 0.0, 90.0, 0.0, 0.01).throttle
    alt_ctrl.reset()
    tilted = alt_ctrl.update(10.0, 0.0, 10.0, 0.0, 90.0, np.radians(20), 0.01).throttle
    assert tilted == pytest.approx(flat / np.cos(np.radians(20)), rel=1e-9)


def test_feedforward_uses_the_estimated_mass(alt_ctrl):
    heavy = alt_ctrl.update(10.0, 0.0, 10.0, 0.0, 100.0, 0.0, 0.01).throttle
    alt_ctrl.reset()
    light = alt_ctrl.update(10.0, 0.0, 10.0, 0.0, 85.0, 0.0, 0.01).throttle
    assert heavy / light == pytest.approx(100.0 / 85.0)


def test_below_target_raises_throttle_and_above_lowers_it(alt_ctrl):
    hover = alt_ctrl.update(10.0, 0.0, 10.0, 0.0, 90.0, 0.0, 0.01).throttle
    alt_ctrl.reset()
    low = alt_ctrl.update(9.0, 0.0, 10.0, 0.0, 90.0, 0.0, 0.01).throttle
    alt_ctrl.reset()
    high = alt_ctrl.update(11.0, 0.0, 10.0, 0.0, 90.0, 0.0, 0.01).throttle
    assert high < hover < low


def test_velocity_mode_ignores_altitude_and_tracks_vz_ref(alt_ctrl):
    # alt_ref=None: only the vertical velocity is regulated
    hover = alt_ctrl.update(3.0, 0.0, None, 0.0, 90.0, 0.0, 0.01).throttle
    alt_ctrl.reset()
    sinking_too_fast = alt_ctrl.update(3.0, -2.0, None, -0.5, 90.0, 0.0, 0.01).throttle
    assert sinking_too_fast > hover


def test_throttle_is_clamped_to_the_engine_range(alt_ctrl, params):
    top = alt_ctrl.update(0.0, -10.0, 100.0, 0.0, 90.0, 0.0, 0.01)
    assert top.throttle == 1.0 and top.saturated
    alt_ctrl.reset()
    bottom = alt_ctrl.update(100.0, 10.0, 0.0, 0.0, 90.0, 0.0, 0.01)
    assert bottom.throttle == pytest.approx(params.thrust_min_frac) and bottom.saturated


def test_cut_command_gives_zero_throttle(alt_ctrl):
    assert alt_ctrl.update(10.0, 0.0, 10.0, 0.0, 90.0, 0.0, 0.01, cut=True).throttle == 0.0


def test_integrator_does_not_wind_up_while_saturated(alt_ctrl):
    for _ in range(500):  # 5 s pinned at the minimum throttle
        alt_ctrl.update(100.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.01)
    assert abs(alt_ctrl.integral) < 1e-6


def test_integrator_removes_a_constant_acceleration_offset(alt_ctrl):
    # Plant: vz' = a_offset + (thrust accel - g); the unmodelled offset must be integrated out.
    thr, z, vz, mass = None, 10.0, 0.0, 90.0
    for _ in range(4000):
        thr = alt_ctrl.update(z, vz, 10.0, 0.0, mass, 0.0, 0.01).throttle
        a = thr * 2500.0 / mass - G - 0.5
        vz += a * 0.01
        z += vz * 0.01
    assert z == pytest.approx(10.0, abs=0.01)


# --- horizontal: unit ---------------------------------------------------------------------


def tilt_of(q):
    return float(np.arccos(np.clip(quat_rotate(q, UP)[2], -1, 1)))


def test_no_error_commands_upright_attitude(hor_ctrl):
    out = hor_ctrl.update(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), 0.0, 0.01)
    assert tilt_of(out.q_des) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(out.a_des, 0.0)


def test_target_east_tilts_thrust_toward_east(hor_ctrl):
    out = hor_ctrl.update(np.zeros(2), np.zeros(2), np.array([2.0, 0.0]), np.zeros(2), 0.0, 0.01)
    thrust_dir = quat_rotate(out.q_des, UP)
    assert thrust_dir[0] > 0.0 and thrust_dir[1] == pytest.approx(0.0, abs=1e-12)


def test_target_north_tilts_thrust_toward_north(hor_ctrl):
    out = hor_ctrl.update(np.zeros(2), np.zeros(2), np.array([0.0, 2.0]), np.zeros(2), 0.0, 0.01)
    thrust_dir = quat_rotate(out.q_des, UP)
    assert thrust_dir[1] > 0.0 and thrust_dir[0] == pytest.approx(0.0, abs=1e-12)


def test_velocity_error_alone_commands_a_tilt_opposing_the_velocity(hor_ctrl):
    out = hor_ctrl.update(np.zeros(2), np.array([1.0, 0.0]), np.zeros(2), np.zeros(2), 0.0, 0.01)
    assert quat_rotate(out.q_des, UP)[0] < 0.0


def test_acceleration_command_is_pd_law(hor_ctrl):
    g = hor_ctrl.gains
    out = hor_ctrl.update(
        np.array([1.0, 0.0]), np.array([0.5, 0.0]), np.array([1.4, 0.0]), np.zeros(2), 0.0, 0.01
    )
    assert out.a_des[0] == pytest.approx(g.kp * 0.4 - g.kd * 0.5, rel=1e-9)


def test_tilt_is_clamped(hor_ctrl):
    out = hor_ctrl.update(np.zeros(2), np.zeros(2), np.array([500.0, 0.0]), np.zeros(2), 0.0, 10.0)
    assert tilt_of(out.q_des) == pytest.approx(HorizontalGains().max_tilt, rel=1e-9)


def test_setpoint_is_rate_limited(hor_ctrl):
    dt = 0.01
    hor_ctrl.update(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), 0.0, dt)
    out = hor_ctrl.update(np.zeros(2), np.zeros(2), np.array([500.0, 0.0]), np.zeros(2), 0.0, dt)
    assert tilt_of(out.q_des) == pytest.approx(HorizontalGains().tilt_rate_max * dt, rel=1e-6)


def test_yaw_setpoint_is_reflected_in_q_des(hor_ctrl):
    yaw = 0.7
    out = hor_ctrl.update(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), yaw, 0.01)
    x_axis = quat_rotate(out.q_des, np.array([1.0, 0.0, 0.0]))
    assert np.arctan2(x_axis[1], x_axis[0]) == pytest.approx(yaw, abs=1e-12)


def test_q_des_is_unit_norm(hor_ctrl):
    out = hor_ctrl.update(np.zeros(2), np.zeros(2), np.array([3.0, -4.0]), np.zeros(2), 1.0, 1.0)
    assert np.linalg.norm(out.q_des) == pytest.approx(1.0, abs=1e-12)


def test_position_loop_is_slower_than_the_attitude_loop():
    g = HorizontalGains()
    wn_pos = np.sqrt(g.kp)
    wn_att = AttitudeGains.from_bandwidth(3.0, 0.8).kp[0] ** 0.5
    assert wn_att / wn_pos >= 3.0


# --- closed loop --------------------------------------------------------------------------


class SetpointController:
    """Test harness: a PositionController tracking a fixed setpoint."""

    def __init__(self, params, setpoint_at):
        self.pos = PositionController(params)
        self.setpoint_at = setpoint_at
        self.last_t = None

    def reset(self):
        self.pos.reset()
        self.last_t = None

    def step(self, sensors, t):
        raise NotImplementedError

    def step_state(self, est, t):
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        return self.pos.command(est, self.setpoint_at(t), dt)


def hover_x0(cfg, alt, xy=(0.0, 0.0)):
    com = mass_properties(PROP_MASS, cfg).com_z
    z = alt + com + cfg.legs.length
    hover_thr = (cfg.mass.dry_mass + PROP_MASS) * G / cfg.engine.thrust_max
    return make_state([xy[0], xy[1], z], [0, 0, 0], None, [0, 0, 0], PROP_MASS, thr=hover_thr)


def fly(cfg, params, setpoint_at, x0, t_end, delay=0.0, env=None):
    ctrl = SetpointController(params, setpoint_at)
    sim = Simulation(
        cfg,
        ctrl,
        env or Environment.default(),
        x0,
        np.random.default_rng(0),
        command_delay=delay,
        log_decimation=10,
    )
    return sim.run(t_end).log


def foot_alt(log, cfg):
    """Foot-tip height from the logged CoM height and propellant (the CoM moves as it burns)."""
    com = np.array([mass_properties(p, cfg).com_z for p in log["prop"]])
    return log["pos_z"].to_numpy() - com - cfg.legs.length


def test_position_controller_satisfies_state_feedback_protocol(params):
    assert isinstance(SetpointController(params, lambda t: Setpoint()), StateFeedbackController)


def test_altitude_step_of_5m_settles_within_1_percent_with_small_overshoot(cfg, params):
    sp = Setpoint(alt_ref=15.0, vz_ref=0.0, xy_ref=np.zeros(2))
    log = fly(cfg, params, lambda t: sp, hover_x0(cfg, 10.0), 15.0)
    alt = foot_alt(log, cfg)
    assert alt.max() <= 10.0 + 5.0 * 1.15
    last = alt[log.t >= 12.0]
    assert np.all(np.abs(last - 15.0) < 0.05)  # 1% of the 5 m step


def test_horizontal_step_respects_tilt_clamp_and_keeps_altitude(cfg, params):
    sp = Setpoint(alt_ref=10.0, vz_ref=0.0, xy_ref=np.array([5.0, 0.0]))
    log = fly(cfg, params, lambda t: sp, hover_x0(cfg, 10.0), 20.0)
    q = log[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy()
    tilt = np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
    # the clamp bounds the command; allow 1.5 degrees of tracking overshoot of the attitude loop
    assert tilt.max() <= HorizontalGains().max_tilt + np.radians(1.5)
    alt = foot_alt(log, cfg)
    assert np.abs(alt - 10.0).max() < 1.0
    last = log[log.t >= 17.0]
    assert np.all(np.abs(last["pos_x"] - 5.0) < 0.1)
    assert np.all(np.abs(last["pos_y"]) < 0.1)


def test_hover_with_20ms_delay_holds_position(cfg, params):
    sp = Setpoint(alt_ref=10.0, vz_ref=0.0, xy_ref=np.zeros(2))
    log = fly(cfg, params, lambda t: sp, hover_x0(cfg, 10.0), 8.0, delay=0.02)
    alt = foot_alt(log, cfg)
    assert np.abs(alt - 10.0).max() < 0.3
    assert np.hypot(log["pos_x"], log["pos_y"]).max() < 0.3


def test_command_type(cfg, params):
    pc = PositionController(params)
    q = np.array([1.0, 0, 0, 0])
    est = StateEstimate(0.0, np.array([0, 0, 12.0]), np.zeros(3), q, np.zeros(3), 90.0)
    cmd = pc.command(est, Setpoint(alt_ref=10.0), 0.0)
    assert isinstance(cmd, Command)
