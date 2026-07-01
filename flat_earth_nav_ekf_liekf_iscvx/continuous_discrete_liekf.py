"""Continuous-discrete right-invariant EKF for flat-earth navigation.

This implements the paper's RIEKF/IEKF on SE_2(3) for landmark observations.
The state update is chi_hat^+ = Exp(delta) chi_hat.
"""

from __future__ import annotations

import numpy as np

from dynamics import (
    State,
    block_diag_rotation_noise,
    iekf_A_matrix,
    iekf_H_matrix,
    iekf_Q_hat,
    iekf_residual,
    propagate_state,
    right_retract,
)
from integrator import covariance_euler, spd_project


class ContinuousDiscreteNavLIEKF:
    """Paper-style invariant EKF with right-invariant SE_2(3) error."""

    def __init__(self, state0: State, P0, Q_base, obs_cov, dt: float, landmarks, gravity) -> None:
        self.state = state0.copy()
        self.P = np.asarray(P0, dtype=float).reshape(9, 9).copy()
        self.Q_base = np.asarray(Q_base, dtype=float).reshape(9, 9)
        self.obs_cov = np.asarray(obs_cov, dtype=float).reshape(3, 3)
        self.dt = float(dt)
        self.landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 3)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.H = iekf_H_matrix(self.landmarks)

    def predict(self, omega) -> None:
        A = iekf_A_matrix(self.gravity)
        Qhat = iekf_Q_hat(self.state, self.Q_base)
        self.state = propagate_state(self.state, omega, self.dt, self.gravity)
        self.P = covariance_euler(self.P, A, Qhat, self.dt)
        self.P = spd_project(self.P, 1e-14)

    def update(self, y) -> None:
        residual = iekf_residual(self.state, y, self.landmarks)
        Nhat = block_diag_rotation_noise(self.state, self.obs_cov, len(self.landmarks))
        S = self.H @ self.P @ self.H.T + Nhat
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        delta = K @ residual
        self.state = right_retract(self.state, delta)
        I = np.eye(9)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ Nhat @ K.T
        self.P = spd_project(self.P, 1e-14)

    def step(self, omega, y=None) -> State:
        self.predict(omega)
        if y is not None:
            self.update(y)
        return self.state.copy()
