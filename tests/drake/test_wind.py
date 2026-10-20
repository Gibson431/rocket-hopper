import numpy as np
from pydrake.all import DiagramBuilder, Simulator

from hopper.drake.wind import WindSource


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
