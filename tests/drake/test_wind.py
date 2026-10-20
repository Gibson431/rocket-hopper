import numpy as np
from pydrake.all import DiagramBuilder, Simulator

from hopper.drake.plant import build_vehicle_plant
from hopper.drake.sim import make_simulator
from hopper.drake.wind import WindSource
from hopper.dynamics.environment import Wind


def _run(seed, sigma=2.0, mean=(1.0, -2.0, 0.5), t_end=1.0, chunks=1, tau=3.0):
    builder = DiagramBuilder()
    wind = builder.AddSystem(WindSource(np.array(mean), sigma, tau, seed, dt=1e-3))
    diagram = builder.Build()
    sim = Simulator(diagram)
    ctx = sim.get_mutable_context()
    out = []
    for k in range(1, chunks + 1):
        sim.AdvanceTo(t_end * k / chunks)
        out.append(wind.get_wind_output_port().Eval(wind.GetMyContextFromRoot(ctx)).copy())
    return np.array(out)


def test_same_seed_gives_bit_identical_sequences():
    assert np.array_equal(_run(7, chunks=5), _run(7, chunks=5))


def test_different_seeds_differ():
    assert not np.array_equal(_run(7), _run(8))


def test_zero_sigma_gives_exactly_the_mean_wind():
    w = _run(3, sigma=0.0, chunks=4)
    assert np.array_equal(w, np.tile([1.0, -2.0, 0.5], (4, 1)))


def test_gust_depends_on_time_only_not_on_how_the_run_is_advanced():
    a, b = _run(5, chunks=1), _run(5, chunks=20)
    assert np.array_equal(a[-1], b[-1])


def test_gust_is_stationary_ou_with_requested_sigma():
    # sample std of the gust over many correlation times is near sigma
    builder = DiagramBuilder()
    wind = builder.AddSystem(WindSource(np.zeros(3), 2.0, 0.5, 11, dt=1e-2))
    sim = Simulator(builder.Build())
    ctx = sim.get_mutable_context()
    samples = []
    for k in range(1, 4001):
        sim.AdvanceTo(0.05 * k)
        samples.append(wind.get_wind_output_port().Eval(wind.GetMyContextFromRoot(ctx)).copy())
    assert abs(np.std(np.array(samples)) - 2.0) < 0.2


def test_gust_sequence_matches_the_legacy_wind_discretisation():
    mean, sigma, tau, dt = np.array([1.0, -2.0, 0.5]), 2.0, 3.0, 1e-3
    legacy = Wind(mean, sigma, tau, None)
    legacy.reset(7)
    for _ in range(500):
        legacy.advance(dt)
    # sample between updates so the event at exactly t = 0.5 s is not a boundary case
    assert np.array_equal(_run(7, t_end=0.5005)[-1], legacy.velocity(0.5, np.zeros(3)))


def test_same_system_replays_the_same_sequence_in_a_second_simulator():
    builder = DiagramBuilder()
    wind = builder.AddSystem(WindSource(np.zeros(3), 2.0, 3.0, 4))
    diagram = builder.Build()
    outs = []
    for _ in range(2):
        sim = Simulator(diagram)
        sim.AdvanceTo(0.3)
        ctx = wind.GetMyContextFromRoot(sim.get_context())
        outs.append(wind.get_wind_output_port().Eval(ctx).copy())
    assert np.array_equal(outs[0], outs[1])


def test_gust_is_the_same_under_different_integrators_and_repeated_evaluation(cfg):
    results = []
    for fixed in (True, False):
        builder = DiagramBuilder()
        plant, _ = build_vehicle_plant(cfg, builder)
        wind = builder.AddSystem(WindSource(np.zeros(3), 2.0, 3.0, 9))
        diagram = builder.Build()
        sim = make_simulator(diagram, dt=1e-3) if fixed else Simulator(diagram)
        sim.AdvanceTo(0.2)
        ctx = wind.GetMyContextFromRoot(sim.get_context())
        first = wind.get_wind_output_port().Eval(ctx).copy()
        assert np.array_equal(first, wind.get_wind_output_port().Eval(ctx))
        results.append(first)
    assert np.array_equal(results[0], results[1])
