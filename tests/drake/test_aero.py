import numpy as np
import pytest
from pydrake.all import ConstantVectorSource, DiagramBuilder, Simulator

from hopper.drake.aero import AeroLoads
from hopper.drake.combiner import ForceCombiner
from hopper.drake.plant import build_vehicle_plant
from hopper.drake.sim import make_simulator
from hopper.drake.state import set_legacy_state, to_legacy_state
from hopper.drake.wind import WindSource
from hopper.dynamics.aero import aero_loads
from hopper.dynamics.environment import atmosphere
from hopper.dynamics.mass_model import mass_properties
from hopper.types import make_state


def _eval(cfg, x, wind=(0.0, 0.0, 0.0), enabled=True, prop=None):
    builder = DiagramBuilder()
    plant, _ = build_vehicle_plant(cfg, builder)
    aero = builder.AddSystem(AeroLoads(cfg, plant, aero_enabled=enabled))
    wsrc = builder.AddSystem(ConstantVectorSource(np.asarray(wind, dtype=float)))
    psrc = builder.AddSystem(
        ConstantVectorSource([cfg.tank.prop_mass_initial if prop is None else prop])
    )
    builder.Connect(plant.get_state_output_port(), aero.get_plant_state_input_port())
    builder.Connect(wsrc.get_output_port(), aero.get_wind_input_port())
    builder.Connect(psrc.get_output_port(), aero.get_propellant_input_port())
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    set_legacy_state(plant, plant.GetMyMutableContextFromRoot(ctx), x, cfg)
    return aero.get_force_output_port().Eval(aero.GetMyContextFromRoot(ctx))[0]


def _state(pos, vel=(0, 0, 0), omega=(0, 0, 0), quat=(1, 0, 0, 0)):
    return make_state(pos=pos, vel=vel, quat=np.array(quat, float), omega=omega, prop=20.0)


def _moving_tilted():
    q = np.array([0.95, 0.1, 0.2, 0.2])
    return _state([0, 0, 30.0], [8.0, -3.0, 12.0], [0.1, 0.2, 0.3], q)


def test_aero_gives_finite_loads_for_a_moving_tilted_vehicle_in_wind(cfg):
    f = _eval(cfg, _moving_tilted(), wind=(4.0, 1.0, 0.0)).F_Bq_W
    assert np.all(np.isfinite(f.translational())) and np.all(np.isfinite(f.rotational()))
    assert np.linalg.norm(f.translational()) > 0.0


def test_aero_disabled_gives_zero_load(cfg):
    f = _eval(cfg, _moving_tilted(), wind=(4.0, 1.0, 0.0), enabled=False).F_Bq_W
    assert np.all(f.translational() == 0.0) and np.all(f.rotational() == 0.0)


def test_aero_force_matches_quasi_steady_drag_law(cfg):
    x = _moving_tilted()
    wind = np.array([4.0, 1.0, 0.0])
    mp = mass_properties(cfg.tank.prop_mass_initial, cfg)
    f_expected, _ = aero_loads(x[3:6], x[6:10], wind, atmosphere(x[2]).density, mp, cfg.aero)
    out = _eval(cfg, x, wind=wind)
    assert out.F_Bq_W.translational() == pytest.approx(f_expected, rel=1e-9, abs=1e-9)
    # applied at the centre of pressure, on the body z axis from the pivot
    assert out.p_BoBq_B == pytest.approx([0.0, 0.0, cfg.aero.cp_z])


def test_air_at_rest_relative_to_the_vehicle_gives_no_force(cfg):
    x = _state([0, 0, 10.0], vel=[3.0, 0.0, 0.0])
    f = _eval(cfg, x, wind=(3.0, 0.0, 0.0)).F_Bq_W
    assert np.allclose(f.translational(), 0.0, atol=1e-12)


def test_rotation_about_cp_drag_uses_com_velocity_not_pivot_velocity(cfg):
    # pure spin about the CoM: CoM velocity is zero, so no drag
    x = _state([0, 0, 10.0], omega=[0.0, 2.0, 0.0])
    f = _eval(cfg, x).F_Bq_W
    assert np.allclose(f.translational(), 0.0, atol=1e-9)


def test_force_combiner_concatenates_forces_from_all_inputs(cfg):
    comb = ForceCombiner(2)
    ctx = comb.CreateDefaultContext()
    from pydrake.all import AbstractValue, ExternallyAppliedSpatialForce

    for i in range(2):
        f = ExternallyAppliedSpatialForce()
        comb.get_input_port(i).FixValue(ctx, AbstractValue.Make([f]))
    assert len(comb.get_output_port().Eval(ctx)) == 2


def test_gravity_can_be_switched_off(cfg):
    for enabled, expect in ((True, -9.80665), (False, 0.0)):
        builder = DiagramBuilder()
        plant, _ = build_vehicle_plant(cfg, builder, gravity_enabled=enabled)
        diagram = builder.Build()
        sim = make_simulator(diagram, dt=1e-3)
        ctx = sim.get_mutable_context()
        set_legacy_state(plant, plant.GetMyMutableContextFromRoot(ctx), _state([0, 0, 5.0]), cfg)
        sim.AdvanceTo(0.5)
        x = to_legacy_state(plant, plant.GetMyContextFromRoot(sim.get_context()))
        assert x[5] == pytest.approx(0.5 * expect, abs=1e-9)


def test_full_diagram_with_wind_and_aero_runs_finite(cfg):
    builder = DiagramBuilder()
    plant, _ = build_vehicle_plant(cfg, builder)
    aero = builder.AddSystem(AeroLoads(cfg, plant))
    wind = builder.AddSystem(WindSource(np.array([3.0, 0.0, 0.0]), 1.0, 2.0, 1))
    psrc = builder.AddSystem(ConstantVectorSource([cfg.tank.prop_mass_initial]))
    comb = builder.AddSystem(ForceCombiner(1))
    builder.Connect(plant.get_state_output_port(), aero.get_plant_state_input_port())
    builder.Connect(wind.get_wind_output_port(), aero.get_wind_input_port())
    builder.Connect(psrc.get_output_port(), aero.get_propellant_input_port())
    builder.Connect(aero.get_force_output_port(), comb.get_input_port(0))
    builder.Connect(comb.get_output_port(), plant.get_applied_spatial_force_input_port())
    sim = Simulator(builder.Build())
    ctx = sim.get_mutable_context()
    set_legacy_state(plant, plant.GetMyMutableContextFromRoot(ctx), _moving_tilted(), cfg)
    sim.AdvanceTo(0.5)
    assert np.all(np.isfinite(plant.GetPositionsAndVelocities(plant.GetMyContextFromRoot(ctx))))
