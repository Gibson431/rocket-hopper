"""Quaternion and rotation helpers. Scalar-first [w, x, y, z], body -> inertial."""

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Rotation matrix rot with v_I = rot @ v_B."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def dcm_to_quat(rot: np.ndarray) -> np.ndarray:
    """Shepperd's method: branch on the largest diagonal term for numerical stability."""
    tr = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if tr >= max(rot[0, 0], rot[1, 1], rot[2, 2]):
        s = 2.0 * np.sqrt(1.0 + tr)
        q = np.array(
            [
                0.25 * s,
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
            ]
        )
    elif rot[0, 0] >= max(rot[1, 1], rot[2, 2]):
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        q = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                0.25 * s,
                (rot[0, 1] + rot[1, 0]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
            ]
        )
    elif rot[1, 1] >= rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        q = np.array(
            [
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[0, 1] + rot[1, 0]) / s,
                0.25 * s,
                (rot[1, 2] + rot[2, 1]) / s,
            ]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        q = np.array(
            [
                (rot[1, 0] - rot[0, 1]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
                (rot[1, 2] + rot[2, 1]) / s,
                0.25 * s,
            ]
        )
    return quat_normalize(q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Body -> inertial."""
    return quat_to_dcm(q) @ v


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Inertial -> body."""
    return quat_to_dcm(q).T @ v


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate(([np.cos(angle / 2)], np.sin(angle / 2) * axis))


def quat_from_euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """Return [yaw, pitch, roll]. Display only; singular at pitch = +-90 deg."""
    rot = quat_to_dcm(q)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    pitch = -np.arcsin(np.clip(rot[2, 0], -1.0, 1.0))
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    return np.array([yaw, pitch, roll])


def quat_error(q_des: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Body-frame rotation vector taking q to q_des (q_des = q (x) dq), shortest path."""
    dq = quat_mul(quat_conj(q), q_des)
    if dq[0] < 0:
        dq = -dq
    v = dq[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2.0 * v
    return 2.0 * np.arctan2(n, dq[0]) * v / n


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    return 0.5 * quat_mul(q, np.concatenate(([0.0], omega_body)))


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
