"""Continuous-discrete LIEKF for the simplified car with GPS output.

This implements the paper's left-invariant GPS case:

    chi_hat^+ = chi_hat exp(L r),
    r = R(theta_hat)^T (y_gps - x_hat).
"""

from __future__ import annotations

import numpy as np

from dynamics import (
    SE2_to_pose,
    gps_H_matrix,
    liekf_A_matrix,
    liekf_left_gps_residual,
    pose_to_SE2,
    propagate_pose_rk4,
    rot2,
    se2_exp,
    wrap_angle,
)
from integrator import covariance_euler


class ContinuousDiscreteCarLIEKF:
    """Left-invariant EKF with GPS/position output."""

    def __init__(self, z0, P0, Q, N, dt: float) -> None:
        self.z = np.asarray(z0, dtype=float).reshape(3).copy()
        self.z[0] = wrap_angle(self.z[0])
        self.P = np.asarray(P0, dtype=float).reshape(3, 3).copy()
        self.Q = np.asarray(Q, dtype=float).reshape(3, 3)
        self.N = np.asarray(N, dtype=float).reshape(2, 2)
        self.dt = float(dt)
        self.H = gps_H_matrix()

    def predict(self, u) -> None:
        u = np.asarray(u, dtype=float).reshape(2)
        v, omega = float(u[0]), float(u[1])
        A = liekf_A_matrix(v, omega)
        self.z = propagate_pose_rk4(self.z, u, self.dt)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P += 1e-15 * np.eye(3)

    def update(self, y_gps) -> None:
        y_gps = np.asarray(y_gps, dtype=float).reshape(2)
        residual = liekf_left_gps_residual(self.z, y_gps)

        Rhat = rot2(float(self.z[0]))
        Nhat = Rhat.T @ self.N @ Rhat
        S = self.H @ self.P @ self.H.T + Nhat
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T

        delta = K @ residual
        chi_plus = pose_to_SE2(self.z) @ se2_exp(delta)
        self.z = SE2_to_pose(chi_plus)

        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def step(self, u, y_gps=None) -> np.ndarray:
        self.predict(u)
        if y_gps is not None:
            self.update(y_gps)
        return self.z.copy()
