"""Continuous-discrete multiplicative EKF for flat-earth navigation."""

from __future__ import annotations

import numpy as np

from dynamics import (
    State,
    ekf_A_matrix,
    ekf_H_matrix,
    measurement_model,
    propagate_state,
    standard_retract,
)
from integrator import covariance_euler, spd_project


class ContinuousDiscreteNavEKF:
    """State-of-the-art MEKF baseline on SO(3) x R^6."""

    def __init__(self, state0: State, P0, Q, N, dt: float, landmarks, gravity) -> None:
        self.state = state0.copy()
        self.P = np.asarray(P0, dtype=float).reshape(9, 9).copy()
        self.Q = np.asarray(Q, dtype=float).reshape(9, 9)
        self.N = np.asarray(N, dtype=float)
        self.dt = float(dt)
        self.landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 3)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)

    def predict(self, omega) -> None:
        A = ekf_A_matrix(self.state, omega)
        self.state = propagate_state(self.state, omega, self.dt, self.gravity)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P = spd_project(self.P, 1e-14)

    def update(self, y) -> None:
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = measurement_model(self.state, self.landmarks)
        residual = y - y_pred
        H = ekf_H_matrix(self.state, self.landmarks)
        S = H @ self.P @ H.T + self.N
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        delta = K @ residual
        self.state = standard_retract(self.state, delta)
        I = np.eye(9)
        # Joseph form for numerical robustness.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.N @ K.T
        self.P = spd_project(self.P, 1e-14)

    def step(self, omega, y=None) -> State:
        self.predict(omega)
        if y is not None:
            self.update(y)
        return self.state.copy()
