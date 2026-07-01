"""Continuous-discrete standard EKF for the simplified car with GPS output."""

from __future__ import annotations

import numpy as np

from dynamics import ekf_A_matrix, gps_H_matrix, propagate_pose_rk4, wrap_angle
from integrator import covariance_euler


class ContinuousDiscreteCarEKF:
    """Standard coordinate EKF from the paper's Sec. IV-D."""

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
        v = float(u[0])
        theta_hat = float(self.z[0])
        A = ekf_A_matrix(theta_hat, v)
        self.z = propagate_pose_rk4(self.z, u, self.dt)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P += 1e-15 * np.eye(3)

    def update(self, y_gps) -> None:
        y_gps = np.asarray(y_gps, dtype=float).reshape(2)
        residual = y_gps - self.z[1:3]
        S = self.H @ self.P @ self.H.T + self.N
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T

        dz = K @ residual
        self.z = self.z + dz
        self.z[0] = wrap_angle(self.z[0])

        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def step(self, u, y_gps=None) -> np.ndarray:
        self.predict(u)
        if y_gps is not None:
            self.update(y_gps)
        return self.z.copy()
