import numpy as np
from hopper_core.types import POS, PROP, QUAT, VEL, make_state
from pydrake.all import DiagramBuilder

from hopper.drake.mass import MassUpdater
from hopper.drake.plant import build_vehicle_plant
from hopper.drake.sim import make_simulator
from hopper.drake.state import set_state_vector, to_state_vector
from hopper.dynamics.mass_model import mass_properties


def _build(cfg):
    builder = DiagramBuilder()
    plant, scene_graph = build_vehicle_plant(cfg, builder)
    return builder.Build(), plant, scene_graph


def test_plant_builds_and_is_finalised(cfg):
    _, plant, scene_graph = _build(cfg)
    assert plant.is_finalized()
    assert scene_graph is not None
    assert plant.num_positions() == 7 and plant.num_velocities() == 6
    assert (
        plant.GetBodyByName("hopper").default_mass()
        == cfg.mass.dry_mass + cfg.tank.prop_mass_initial
    )


def test_plant_gravity_is_standard_gravity(cfg):
    _, plant, _ = _build(cfg)
    assert plant.gravity_field().gravity_vector()[2] == -9.80665


def test_one_second_of_free_fall_is_finite_and_ballistic(cfg):
    diagram, plant, _ = _build(cfg)
    sim = make_simulator(diagram, dt=1e-3)
    ctx = sim.get_mutable_context()
    pctx = plant.GetMyMutableContextFromRoot(ctx)
    x0 = make_state(
        pos=[0, 0, 50.0],
        vel=[1.0, 0.0, 2.0],
        quat=None,
        omega=[0, 0, 0],
        prop=cfg.tank.prop_mass_initial,
    )
    set_state_vector(plant, pctx, x0, cfg)
    sim.AdvanceTo(1.0)
    x = to_state_vector(plant, pctx)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x[POS], [1.0, 0.0, 50.0 + 2.0 - 0.5 * 9.80665], atol=1e-9)
    np.testing.assert_allclose(x[VEL], [1.0, 0.0, 2.0 - 9.80665], atol=1e-9)


def test_simulator_uses_fixed_step_rk3(cfg):
    diagram, _, _ = _build(cfg)
    sim = make_simulator(diagram, dt=1e-3)
    integ = sim.get_integrator()
    assert type(integ).__name__ == "RungeKutta3Integrator"
    assert integ.get_fixed_step_mode()
    assert integ.get_maximum_step_size() == 1e-3


def test_free_fall_is_deterministic(cfg):
    out = []
    for _ in range(2):
        diagram, plant, _ = _build(cfg)
        sim = make_simulator(diagram)
        pctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
        set_state_vector(
            plant,
            pctx,
            make_state(
                pos=[0, 0, 9.0],
                vel=[0.3, 0.1, 0.0],
                quat=[0.9, 0.1, -0.2, 0.3],
                omega=[0.1, 0.2, 0.3],
                prop=10.0,
            ),
            cfg,
        )
        sim.AdvanceTo(0.5)
        out.append(to_state_vector(plant, pctx))
    assert np.array_equal(out[0], out[1])


def test_mass_updater_writes_mass_com_and_inertia(cfg):
    _, plant, _ = _build(cfg)
    ctx = plant.CreateDefaultContext()
    body = plant.GetBodyByName("hopper")
    upd = MassUpdater(plant, cfg)
    for prop in (20.0, 0.0, 7.3):
        upd.update(ctx, prop)
        mp = mass_properties(prop, cfg)
        si = body.CalcSpatialInertiaInBodyFrame(ctx)
        assert si.get_mass() == mp.mass
        np.testing.assert_allclose(si.get_com(), [0.0, 0.0, mp.com_z], atol=1e-12)
        central = si.Shift(si.get_com()).CalcRotationalInertia().CopyToFullMatrix3()
        # Shift(p) gives the inertia about p; shifting to the CoM recovers the central inertia
        np.testing.assert_allclose(central, mp.inertia, atol=1e-9)


def test_mass_update_keeps_com_state_continuous(cfg):
    _, plant, _ = _build(cfg)
    ctx = plant.CreateDefaultContext()
    x = make_state(
        pos=[1, 2, 3],
        vel=[0.5, -0.5, 1.0],
        quat=[0.9, 0.1, -0.2, 0.3],
        omega=[0.2, -0.1, 0.4],
        prop=20.0,
    )
    set_state_vector(plant, ctx, x, cfg)
    MassUpdater(plant, cfg).update(ctx, 12.0)
    y = to_state_vector(plant, ctx)
    np.testing.assert_allclose(y[POS], x[POS], atol=1e-12)
    np.testing.assert_allclose(y[VEL], x[VEL], atol=1e-12)
    np.testing.assert_allclose(y[QUAT], x[QUAT], atol=1e-12)  # w > 0 here
    assert y[PROP] == 0.0  # propellant is not a plant state
