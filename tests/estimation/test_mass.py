import pytest

from hopper.estimation.mass import PropellantEstimator
from hopper.scenario_glue import ekf_params_from_config

G0 = 9.80665


def make(cfg, **kw):
    p = ekf_params_from_config(cfg)
    return PropellantEstimator(p, **kw), p


def test_constant_throttle_burns_propellant_at_thrust_over_isp_g0(cfg):
    est, p = make(cfg)
    thr = 0.5
    for k in range(1001):  # 10 s at 100 Hz
        est.update(0.01 * k, thr)
    burned = p.prop_mass_initial - (est.mass - p.dry_mass)
    expected = p.thrust_max * thr * (10.0 - p.throttle_tau) / (p.isp * G0)  # first-order lag
    assert burned == pytest.approx(expected, rel=0.02)


def test_throttle_zero_means_engine_off_and_low_requests_are_raised_to_minimum(cfg):
    est, p = make(cfg)
    for k in range(501):
        est.update(0.01 * k, 0.0)
    assert est.mass == pytest.approx(p.dry_mass + p.prop_mass_initial)
    low, _ = make(cfg)
    ref, _ = make(cfg)
    for k in range(501):
        low.update(0.01 * k, 0.05)
        ref.update(0.01 * k, p.thrust_min_frac)
    assert low.mass == pytest.approx(ref.mass)
    assert low.mass < p.dry_mass + p.prop_mass_initial


def test_mass_never_drops_below_dry_mass(cfg):
    est, p = make(cfg)
    for k in range(20001):
        est.update(0.01 * k, 1.0)
    assert est.mass == pytest.approx(p.dry_mass)
