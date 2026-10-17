import numpy as np
import pytest

from hopper.dynamics.mass_model import mass_properties


def test_full_tank_mass_is_dry_plus_propellant(cfg):
    mp = mass_properties(cfg.tank.prop_mass_initial, cfg)
    assert mp.mass == pytest.approx(cfg.mass.dry_mass + cfg.tank.prop_mass_initial, abs=1e-12)


def test_empty_tank_equals_dry_properties_exactly(cfg):
    mp = mass_properties(0.0, cfg)
    assert mp.mass == cfg.mass.dry_mass
    assert mp.com_z == cfg.mass.dry_com_z
    assert np.array_equal(mp.inertia, np.diag(cfg.mass.dry_inertia))


def test_negative_propellant_is_clamped_to_empty(cfg):
    mp = mass_properties(-1.0, cfg)
    assert mp.mass == cfg.mass.dry_mass
    assert mp.com_z == cfg.mass.dry_com_z


def test_com_lies_between_dry_com_and_tank_centre(cfg):
    tank_centre = cfg.tank.bottom_z + cfg.tank.height / 2
    lo, hi = sorted([cfg.mass.dry_com_z, tank_centre])
    for prop in np.linspace(0.0, cfg.tank.prop_mass_initial, 21):
        assert lo - 1e-12 <= mass_properties(prop, cfg).com_z <= hi + 1e-12


def test_com_rises_monotonically_to_dry_com_over_the_late_burn(cfg):
    # The CoM first dips (the column centre falls faster than the mass shrinks) and turns
    # around at ~13.9 kg for the default vehicle; after that it rises monotonically.
    props = np.linspace(13.0, 0.0, 41)
    com = np.array([mass_properties(p, cfg).com_z for p in props])
    assert np.all(np.diff(com) > 0)
    assert com[-1] == cfg.mass.dry_com_z


def test_com_dips_below_full_tank_value_early_in_the_burn(cfg):
    full = mass_properties(20.0, cfg).com_z
    assert mass_properties(15.0, cfg).com_z < full


def test_inertia_symmetric_positive_definite(cfg):
    for prop in np.linspace(0.0, cfg.tank.prop_mass_initial, 11):
        inertia = mass_properties(prop, cfg).inertia
        assert inertia.shape == (3, 3)
        assert np.allclose(inertia, inertia.T, atol=0, rtol=0)
        assert np.all(np.linalg.eigvalsh(inertia) > 0)


def test_inertia_increases_with_propellant(cfg):
    props = np.linspace(0.0, cfg.tank.prop_mass_initial, 41)
    diag = np.array([np.diag(mass_properties(p, cfg).inertia) for p in props])
    assert np.all(np.diff(diag, axis=0) > 0)


def test_full_tank_inertia_matches_hand_parallel_axis_calculation(cfg):
    # Hand numbers for the default vehicle (dry 80 kg @ 0.9 m, 20 kg tank column
    # r=0.2, h=0.8 centred at 0.7 m): total CoM = (80*0.9 + 20*0.7)/100 = 0.86 m.
    r, h, m_p, m_d = 0.2, 0.8, 20.0, 80.0
    d_dry, d_prop = 0.9 - 0.86, 0.7 - 0.86
    ixx = 28.0 + m_d * d_dry**2 + m_p * (3 * r**2 + h**2) / 12 + m_p * d_prop**2
    izz = 2.5 + m_p * r**2 / 2
    mp = mass_properties(20.0, cfg)
    assert mp.com_z == pytest.approx(0.86, abs=1e-12)
    assert np.allclose(np.diag(mp.inertia), [ixx, ixx, izz], rtol=1e-12)
    assert np.allclose(mp.inertia - np.diag(np.diag(mp.inertia)), 0.0)


def test_half_tank_fills_half_height_from_the_bottom(cfg):
    # 10 kg -> column 0.4 m high, centre at 0.3 + 0.2 = 0.5 m.
    mp = mass_properties(10.0, cfg)
    assert mp.com_z == pytest.approx((80.0 * 0.9 + 10.0 * 0.5) / 90.0, abs=1e-12)
