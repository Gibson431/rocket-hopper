import numpy as np
import pytest
from pydrake.all import ConstantVectorSource, DiagramBuilder

from hopper.drake.plant import build_vehicle_plant
from hopper.drake.propulsion import Engine
from hopper.drake.sim import make_simulator
from hopper.drake.state import set_legacy_state, to_legacy_state
from hopper.types import make_state


def _build(cfg, cmd, prop=None, thrust_frac=0.0, quat=None):
    builder = DiagramBuilder()
    plant, _ = build_vehicle_plant(cfg, builder)
    engine = builder.AddSystem(Engine(cfg, plant))
    src = builder.AddSystem(ConstantVectorSource(np.asarray(cmd, dtype=float)))
    builder.Connect(src.get_output_port(), engine.get_command_input_port())
    builder.Connect(plant.get_state_output_port(), engine.get_plant_state_input_port())
    builder.Connect(engine.get_force_output_port(), plant.get_applied_spatial_force_input_port())
    diagram = builder.Build()
    sim = make_simulator(diagram, dt=1e-3)
    ctx = sim.get_mutable_context()
    prop = cfg.tank.prop_mass_initial if prop is None else prop
    x0 = make_state(pos=[0, 0, 50.0], vel=[0, 0, 0], quat=quat, omega=[0, 0, 0], prop=prop)
    set_legacy_state(plant, plant.GetMyMutableContextFromRoot(ctx), x0, cfg)
    ectx = engine.GetMyMutableContextFromRoot(ctx)
    engine.set_actuator_state(ectx, prop=prop, thrust_frac=thrust_frac)
    return sim, plant, engine, ctx


def test_two_second_burn_is_finite_burns_propellant_and_thrust_stays_in_range(cfg):
    sim, plant, engine, ctx = _build(cfg, [0.8, 0.0, 0.0, 0.0])
    sim.AdvanceTo(2.0)
    ectx = engine.GetMyContextFromRoot(ctx)
    prop = engine.get_propellant_output_port().Eval(ectx)[0]
    thrust = engine.get_thrust_output_port().Eval(ectx)[0]
    x = to_legacy_state(plant, plant.GetMyContextFromRoot(ctx))
    assert np.all(np.isfinite(x)) and np.isfinite(prop)
    assert prop < cfg.tank.prop_mass_initial
    assert (
        cfg.engine.thrust_min_frac * cfg.engine.thrust_max
        <= thrust
        <= 0.8 * cfg.engine.thrust_max + 1e-9
    )
    # thrust 0.8 * 2500 N lifts a ~100 kg vehicle (weight ~981 N)
    assert x[5] > 0.0


def test_positive_gy_pushes_thrust_toward_plus_x_and_torques_about_minus_y(cfg):
    sim, plant, engine, ctx = _build(cfg, [1.0, 0.0, 0.05, 0.0])
    sim.AdvanceTo(0.1)  # short: the torque soon tilts the thrust away
    x = to_legacy_state(plant, plant.GetMyContextFromRoot(ctx))
    assert x[3] > 0.0  # vx
    assert x[11] < 0.0  # omega_y


def test_positive_gx_pushes_thrust_toward_minus_y_and_torques_about_minus_x(cfg):
    sim, plant, engine, ctx = _build(cfg, [1.0, 0.05, 0.0, 0.0])
    sim.AdvanceTo(0.1)  # short: the torque soon tilts the thrust away
    x = to_legacy_state(plant, plant.GetMyContextFromRoot(ctx))
    assert x[4] < 0.0  # vy
    assert x[10] < 0.0  # omega_x


def test_roll_torque_spins_the_vehicle_about_z(cfg):
    sim, plant, engine, ctx = _build(cfg, [0.0, 0.0, 0.0, 1.0])
    sim.AdvanceTo(0.5)
    x = to_legacy_state(plant, plant.GetMyContextFromRoot(ctx))
    assert x[12] > 0.0


def test_empty_tank_gives_zero_thrust_and_propellant_stays_zero(cfg):
    # thrust fraction starts at 1 so only the empty-tank cutoff can make thrust zero
    sim, plant, engine, ctx = _build(cfg, [1.0, 0.0, 0.0, 0.0], prop=0.0, thrust_frac=1.0)
    sim.AdvanceTo(0.5)
    ectx = engine.GetMyContextFromRoot(ctx)
    assert engine.get_thrust_output_port().Eval(ectx)[0] == 0.0
    assert engine.get_propellant_output_port().Eval(ectx)[0] == 0.0
    x = to_legacy_state(plant, plant.GetMyContextFromRoot(ctx))
    assert x[5] < 0.0  # falling: no thrust


def test_engine_off_command_lets_thrust_decay_to_zero(cfg):
    sim, plant, engine, ctx = _build(cfg, [0.0, 0.0, 0.0, 0.0])
    sim.AdvanceTo(0.5)
    ectx = engine.GetMyContextFromRoot(ctx)
    assert engine.get_thrust_output_port().Eval(ectx)[0] == 0.0
    assert engine.get_propellant_output_port().Eval(ectx)[0] == cfg.tank.prop_mass_initial


def test_gimbal_angle_never_exceeds_the_limit(cfg):
    sim, plant, engine, ctx = _build(cfg, [1.0, 1.0, -1.0, 0.0])
    sim.AdvanceTo(1.0)
    g = engine.GetMyContextFromRoot(ctx).get_continuous_state_vector().CopyToVector()[2:4]
    assert np.all(np.abs(g) <= cfg.engine.gimbal_max + 1e-12)
    assert np.all(np.abs(g) > 0.9 * cfg.engine.gimbal_max)


def test_propellant_burn_follows_the_rocket_equation(cfg):
    sim, plant, engine, ctx = _build(cfg, [1.0, 0.0, 0.0, 0.0], thrust_frac=1.0)
    sim.AdvanceTo(2.0)
    ectx = engine.GetMyContextFromRoot(ctx)
    burned = cfg.tank.prop_mass_initial - engine.get_propellant_output_port().Eval(ectx)[0]
    expected = cfg.engine.thrust_max * 2.0 / (cfg.engine.isp * 9.80665)
    assert burned == pytest.approx(expected, rel=1e-6)


def test_low_throttle_request_is_raised_to_the_minimum_and_lag_follows_tau(cfg):
    sim, plant, engine, ctx = _build(cfg, [0.05, 0.0, 0.0, 0.0])
    sim.AdvanceTo(5 * cfg.engine.throttle_tau)
    ectx = engine.GetMyContextFromRoot(ctx)
    frac = engine.get_thrust_output_port().Eval(ectx)[0] / cfg.engine.thrust_max
    expected = cfg.engine.thrust_min_frac * (1 - np.exp(-5))
    assert frac == pytest.approx(expected, rel=1e-3)


def test_thrust_follows_a_tilted_vehicle(cfg):
    # 90 deg about +y: z_B points along +x_I, so full thrust accelerates the vehicle east
    q = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
    sim, plant, engine, ctx = _build(cfg, [1.0, 0.0, 0.0, 0.0], thrust_frac=1.0, quat=q)
    sim.AdvanceTo(0.05)
    x = to_legacy_state(plant, plant.GetMyContextFromRoot(ctx))
    assert x[3] > 1.0


def test_gimbal_rate_stays_within_the_rate_limit(cfg):
    sim, plant, engine, ctx = _build(cfg, [1.0, 1.0, 0.0, 0.0])
    sim.AdvanceTo(0.04)
    rate = engine.GetMyContextFromRoot(ctx).get_continuous_state_vector().CopyToVector()[4]
    assert rate <= 1.05 * cfg.engine.gimbal_rate_max  # state may overshoot one step, as in legacy
    assert rate > 0.9 * cfg.engine.gimbal_rate_max
