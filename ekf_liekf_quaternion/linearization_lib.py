"""Linearization utilities for the quaternion EKF and LIEKF."""

from __future__ import annotations

import numpy as np

from dynamics import measurement_model_left, normalize_quat, skew


def finite_difference_jacobian(fun, x, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Jacobian of fun: R^n -> R^m."""
    x = np.asarray(x, dtype=float).reshape(-1)
    f0 = np.asarray(fun(x), dtype=float).reshape(-1)
    J = np.zeros((f0.size, x.size), dtype=float)
    for j in range(x.size):
        dx = np.zeros_like(x)
        dx[j] = eps
        fp = np.asarray(fun(x + dx), dtype=float).reshape(-1)
        fm = np.asarray(fun(x - dx), dtype=float).reshape(-1)
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J


def measurement_jacobian_quat(q_hat, reference_body_vectors) -> np.ndarray:
    """Ambient 4D measurement Jacobian dh/dq used by the additive EKF."""
    def h(q_raw):
        return measurement_model_left(normalize_quat(q_raw), reference_body_vectors)

    return finite_difference_jacobian(h, np.asarray(q_hat, dtype=float).reshape(4))


def measurement_jacobian_liekf(reference_body_vectors) -> np.ndarray:
    """
    Intrinsic 3D LIEKF measurement Jacobian.

    For r_i = R_hat^T y_i - d_i and eta = R_true^T R_hat = Exp(xi),
    r_i = [d_i]x xi + first-order noise.
    """
    refs = np.asarray(reference_body_vectors, dtype=float).reshape(-1, 3)
    return np.vstack([skew(d) for d in refs])
