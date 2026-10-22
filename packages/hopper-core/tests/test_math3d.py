import numpy as np
import pytest
from hopper_core import math3d as m
from hypothesis import given
from hypothesis import strategies as st

finite = st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False)
unit_quats = (
    st.tuples(finite, finite, finite, finite)
    .filter(lambda t: np.linalg.norm(t) > 0.1)
    .map(lambda t: m.quat_normalize(np.array(t)))
)
vec3 = st.tuples(finite, finite, finite).map(np.array)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


@given(unit_quats)
def test_multiplying_by_identity_leaves_quaternion_unchanged(q):
    assert m.quat_mul(IDENTITY, q) == pytest.approx(q, abs=1e-15)
    assert m.quat_mul(q, IDENTITY) == pytest.approx(q, abs=1e-15)


@given(unit_quats)
def test_conjugate_is_the_inverse_of_a_unit_quaternion(q):
    assert m.quat_mul(q, m.quat_conj(q)) == pytest.approx(IDENTITY, abs=1e-12)


def test_hamilton_product_of_basis_quaternions():
    i = np.array([0.0, 1, 0, 0])
    j = np.array([0.0, 0, 1, 0])
    k = np.array([0.0, 0, 0, 1])
    assert m.quat_mul(i, j) == pytest.approx(k)
    assert m.quat_mul(j, k) == pytest.approx(i)
    assert m.quat_mul(k, i) == pytest.approx(j)


@given(unit_quats)
def test_dcm_is_orthonormal_with_unit_determinant(q):
    rot = m.quat_to_dcm(q)
    assert rot @ rot.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)


@given(unit_quats)
def test_dcm_round_trip_recovers_quaternion_up_to_sign(q):
    q2 = m.dcm_to_quat(m.quat_to_dcm(q))
    assert q2 == pytest.approx(q, abs=1e-9) or q2 == pytest.approx(-q, abs=1e-9)


@pytest.mark.parametrize("angle", [np.pi, -np.pi, 3.0])
def test_dcm_to_quat_handles_near_half_turn_rotations(angle):
    for axis in np.eye(3):
        q = m.quat_from_axis_angle(axis, angle)
        q2 = m.dcm_to_quat(m.quat_to_dcm(q))
        assert min(np.linalg.norm(q2 - q), np.linalg.norm(q2 + q)) < 1e-9


def test_quarter_turn_about_z_maps_x_body_to_y_inertial():
    q = m.quat_from_axis_angle(np.array([0.0, 0, 1]), np.pi / 2)
    assert m.quat_rotate(q, np.array([1.0, 0, 0])) == pytest.approx([0, 1, 0], abs=1e-12)
    assert m.quat_to_dcm(q) @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)


@given(unit_quats, vec3)
def test_rotate_inv_undoes_rotate(q, v):
    assert m.quat_rotate_inv(q, m.quat_rotate(q, v)) == pytest.approx(v, abs=1e-12)


@given(unit_quats, vec3)
def test_rotate_matches_dcm(q, v):
    assert m.quat_rotate(q, v) == pytest.approx(m.quat_to_dcm(q) @ v, abs=1e-12)


def test_axis_angle_normalises_the_axis():
    q = m.quat_from_axis_angle(np.array([0.0, 0, 5.0]), 0.3)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert q == pytest.approx([np.cos(0.15), 0, 0, np.sin(0.15)])


def test_euler_zyx_composes_yaw_then_pitch_then_roll():
    yaw, pitch, roll = 0.3, -0.2, 0.5
    q = m.quat_from_euler_zyx(yaw, pitch, roll)
    ez, ey, ex = (np.eye(3)[i] for i in (2, 1, 0))
    qz = m.quat_from_axis_angle(ez, yaw)
    qy = m.quat_from_axis_angle(ey, pitch)
    qx = m.quat_from_axis_angle(ex, roll)
    expected = m.quat_mul(qz, m.quat_mul(qy, qx))
    assert q == pytest.approx(expected, abs=1e-12)


@given(
    st.floats(-3.0, 3.0),
    st.floats(-1.4, 1.4),
    st.floats(-3.0, 3.0),
)
def test_euler_round_trip(yaw, pitch, roll):
    q = m.quat_from_euler_zyx(yaw, pitch, roll)
    assert m.quat_to_euler_zyx(q) == pytest.approx([yaw, pitch, roll], abs=1e-9)


def test_quat_error_is_zero_for_identical_attitudes():
    q = m.quat_from_euler_zyx(0.4, 0.1, -0.2)
    assert m.quat_error(q, q) == pytest.approx(np.zeros(3), abs=1e-12)


def test_quat_error_takes_the_shortest_path_for_q_and_minus_q():
    q = m.quat_from_euler_zyx(0.4, 0.1, -0.2)
    q_des = m.quat_from_euler_zyx(0.0, 0.0, 0.0)
    assert m.quat_error(q_des, -q) == pytest.approx(m.quat_error(q_des, q), abs=1e-12)
    assert m.quat_error(-q_des, q) == pytest.approx(m.quat_error(q_des, q), abs=1e-12)


@given(
    unit_quats,
    st.tuples(finite, finite, finite).filter(lambda a: np.linalg.norm(a) > 0.1),
    st.floats(-3.0, 3.0),
)
def test_quat_error_has_the_rotation_angle_as_magnitude_in_the_body_frame(q, axis, angle):
    # q_des = q (x) dq, where dq is a body-frame rotation: error must recover angle * axis.
    axis = np.array(axis) / np.linalg.norm(axis)
    q_des = m.quat_mul(q, m.quat_from_axis_angle(axis, angle))
    err = m.quat_error(q_des, q)
    assert np.linalg.norm(err) == pytest.approx(abs(angle), abs=1e-9)
    assert err == pytest.approx(axis * angle, abs=1e-9)


def test_quat_error_is_expressed_in_the_body_frame():
    # Body is yawed 90 deg; desired is body pitched by 0.1 about its own y axis.
    q = m.quat_from_axis_angle(np.array([0.0, 0, 1]), np.pi / 2)
    q_des = m.quat_mul(q, m.quat_from_axis_angle(np.array([0.0, 1, 0]), 0.1))
    assert m.quat_error(q_des, q) == pytest.approx([0, 0.1, 0], abs=1e-12)


def test_quat_derivative_of_constant_omega_integrates_to_the_analytic_quaternion():
    omega = np.array([0.3, -0.5, 0.8])
    q0 = m.quat_from_euler_zyx(0.2, 0.1, -0.3)
    q = q0.copy()
    n, t_end = 20000, 1.0
    dt = t_end / n
    for _ in range(n):
        k1 = m.quat_derivative(q, omega)
        k2 = m.quat_derivative(q + 0.5 * dt * k1, omega)
        k3 = m.quat_derivative(q + 0.5 * dt * k2, omega)
        k4 = m.quat_derivative(q + dt * k3, omega)
        q = m.quat_normalize(q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    w = np.linalg.norm(omega)
    exact = m.quat_mul(q0, m.quat_from_axis_angle(omega / w, w * t_end))
    assert q == pytest.approx(exact, abs=1e-9)


def test_quat_derivative_formula():
    q = IDENTITY
    omega = np.array([2.0, 4.0, 6.0])
    assert m.quat_derivative(q, omega) == pytest.approx([0, 1, 2, 3])


def test_skew_matrix_gives_the_cross_product():
    a, b = np.array([1.0, -2, 3]), np.array([0.5, 4, -1])
    assert m.skew(a) @ b == pytest.approx(np.cross(a, b))
    assert m.skew(a) == pytest.approx(-m.skew(a).T)


def test_quat_normalize_returns_unit_norm():
    assert np.linalg.norm(m.quat_normalize(np.array([2.0, 0, 0, 0]))) == pytest.approx(1.0)


def test_cross_matches_numpy_for_single_and_batched_vectors():
    from hopper_core.math3d import cross

    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
    assert cross(a[0], b[0]) == pytest.approx(np.cross(a[0], b[0]), abs=1e-15)
    assert cross(a, b) == pytest.approx(np.cross(a, b), abs=1e-15)
    assert cross(a, b[0]) == pytest.approx(np.cross(a, b[0]), abs=1e-15)
