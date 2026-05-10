"""
Quaternion dynamics and Lie-group utilities.

Convention
----------
q = [qw, qx, qy, qz] is a unit quaternion that rotates a vector from the
body frame to the inertial frame:

    v_inertial = R(q) v_body = q * [0, v_body] * q_conj.

The continuous attitude dynamics are

    q_dot = 0.5 * q * [0, omega_body],

where omega_body is the body angular velocity.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def skew(v: Array) -> Array:
    """Return the cross-product matrix [v]x such that [v]x w = v x w."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=float,
    )


def normalize_quat(q: Array, keep_positive_scalar: bool = True) -> Array:
    """Normalize a quaternion and optionally use the q ~ -q sign convention."""
    q = np.asarray(q, dtype=float).reshape(4)
    nrm = np.linalg.norm(q)
    if nrm <= 1e-15:
        raise ValueError("Cannot normalize a near-zero quaternion.")
    q = q / nrm
    if keep_positive_scalar and q[0] < 0.0:
        q = -q
    return q


def quat_conj(q: Array) -> Array:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(p: Array, q: Array) -> Array:
    """Hamilton product p * q for scalar-first quaternions."""
    p = np.asarray(p, dtype=float).reshape(4)
    q = np.asarray(q, dtype=float).reshape(4)
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_exp(phi: Array) -> Array:
    """SO(3) exponential map represented as a unit quaternion."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    half_theta = 0.5 * theta
    if theta < 1e-12:
        # sin(theta/2)/theta = 1/2 - theta^2/48 + O(theta^4)
        vec_scale = 0.5 - theta * theta / 48.0
        return normalize_quat(np.r_[1.0 - theta * theta / 8.0, vec_scale * phi])
    axis = phi / theta
    return normalize_quat(np.r_[np.cos(half_theta), np.sin(half_theta) * axis])


def quat_log(q: Array) -> Array:
    """SO(3) logarithm map from a unit quaternion to a rotation vector."""
    q = normalize_quat(q)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return 2.0 * v
    angle = 2.0 * np.arctan2(nv, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return angle * v / nv


def quat_to_rotmat(q: Array) -> Array:
    """Rotation matrix R(q), body frame to inertial frame."""
    q = normalize_quat(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_rotate(q: Array, v: Array) -> Array:
    """Rotate a 3-vector from body frame to inertial frame."""
    return quat_to_rotmat(q) @ np.asarray(v, dtype=float).reshape(3)


def quat_rotate_inverse(q: Array, v: Array) -> Array:
    """Rotate a 3-vector from inertial frame to body frame."""
    return quat_to_rotmat(q).T @ np.asarray(v, dtype=float).reshape(3)


def omega_matrix(omega: Array) -> Array:
    """Matrix Omega(omega) such that q_dot = 0.5 Omega(omega) q."""
    wx, wy, wz = np.asarray(omega, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ],
        dtype=float,
    )


def gyro_input_matrix(q: Array) -> Array:
    """Matrix G(q) such that q_dot = 0.5 * G(q) * omega."""
    qw, qx, qy, qz = np.asarray(q, dtype=float).reshape(4)
    return np.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=float,
    )


def quat_dynamics(qt: Array, omegat: Array) -> Array:
    """Continuous quaternion dynamics."""
    return 0.5 * omega_matrix(omegat) @ np.asarray(qt, dtype=float).reshape(4)


def quat_state_jacobian(omega: Array) -> Array:
    """A_q = d(q_dot)/dq for the additive quaternion EKF."""
    return 0.5 * omega_matrix(omega)


def measurement_model_left(q: Array, reference_body_vectors: Array) -> Array:
    """
    Left-invariant vector observation model.

    Each known body-fixed vector d_i is observed in inertial coordinates:

        y_i = R(q) d_i + noise.

    At least two non-collinear vectors are recommended for full attitude
    observability.
    """
    R = quat_to_rotmat(q)
    refs = np.asarray(reference_body_vectors, dtype=float).reshape(-1, 3)
    return np.concatenate([R @ d for d in refs])


def liekf_left_residual(q_hat: Array, y: Array, reference_body_vectors: Array) -> Array:
    """
    LIEKF invariant residual r_i = R(q_hat)^T y_i - d_i.

    For the left-invariant error eta = R_true^T R_hat = Exp(xi),
    r_i = [d_i]x xi + noise + higher order terms.
    """
    Rhat_T = quat_to_rotmat(q_hat).T
    refs = np.asarray(reference_body_vectors, dtype=float).reshape(-1, 3)
    y = np.asarray(y, dtype=float).reshape(-1, 3)
    return np.concatenate([Rhat_T @ yi - di for yi, di in zip(y, refs)])


def attitude_error_angle(q_true: Array, q_est: Array) -> float:
    """Geodesic attitude error angle in radians."""
    q_true = normalize_quat(q_true)
    q_est = normalize_quat(q_est)
    q_err = quat_mul(quat_conj(q_est), q_true)
    q_err = normalize_quat(q_err)
    return 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0]))
