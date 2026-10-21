"""Filter consistency over seeded runs: NEES of the full error state, NIS of baro and GPS."""

import numpy as np
import pytest
from scipy.stats import chi2

from hopper.math3d import quat_error

from .estimation_harness import Feeder, make_ekf

N_RUNS = 20
T_END = 20.0
N_DOF = 16


def one_run(cfg, seed):
    ekf, feeder = make_ekf(cfg), Feeder(cfg, seed)
    times, nees = [], []
    for t, data, x in feeder:
        if t > T_END:
            break
        est = ekf.process(data)
        if est is None or abs(t / 0.5 - round(t / 0.5)) > 1e-6:
            continue
        bg, ba, bb = feeder.biases()
        err = np.concatenate(
            [
                x[:3] - ekf.p,
                x[3:6] - ekf.v,
                quat_error(x[6:10], ekf.q),
                bg - ekf.bg,
                ba - ekf.ba,
                [bb - ekf.bb],
            ]
        )
        times.append(t)
        nees.append(err @ np.linalg.solve(ekf.P, err))
    return np.array(times), np.array(nees), ekf.nis_log


@pytest.fixture(scope="module")
def runs(cfg_module):
    return [one_run(cfg_module, seed) for seed in range(N_RUNS)]


@pytest.fixture(scope="module")
def cfg_module():
    from hopper.config import load_vehicle

    return load_vehicle("configs/vehicles/vehicle_default.yaml")


def test_average_nees_over_seeded_runs_is_inside_the_chi_square_bounds(runs):
    nees = np.array([r[1] for r in runs])  # (runs, times)
    anees = nees.mean(axis=0)
    lo, hi = chi2.ppf([0.025, 0.975], N_RUNS * N_DOF) / N_RUNS
    inside = ((anees >= lo) & (anees <= hi)).mean()
    print(f"ANEES mean {anees.mean():.2f} (expect {N_DOF}), {inside:.0%} of instants inside 95%")
    assert inside >= 0.8, (inside, anees.round(1), (lo, hi))
    lo99, hi99 = chi2.ppf([0.005, 0.995], N_RUNS * N_DOF * 5) / (N_RUNS * 5)
    assert lo99 < anees.mean() < hi99  # the time average, with a liberal allowance for correlation


@pytest.mark.parametrize(("kind", "dof"), [("baro", 1), ("gps", 6)])
def test_innovations_are_normalised_for_each_sensor(runs, kind, dof):
    nis = np.array([e[2] for _, _, log in runs for e in log if e[0] == kind])
    print(f"{kind}: mean NIS {nis.mean():.2f} over {len(nis)} (expect {dof})")
    assert len(nis) > 500
    lo, hi = chi2.ppf([0.005, 0.995], len(nis) * dof) / len(nis)
    assert lo < nis.mean() < hi, (nis.mean(), dof)
