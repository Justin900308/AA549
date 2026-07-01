"""Continuous-discrete RIEKF for the simplified car with landmark output.

This implements the paper's right-invariant observation case, Sec. IV-B4:

    y_k = R(theta)^T (x - p_k) + V_k,
    chi_hat^+ = exp(L r) chi_hat.

At least two distinct landmarks are needed for full observability.
"""

from __future__ import annotations

import numpy as np

from dynamics import (
    SE2_to_pose,
    landmark_body_measurement_model,
    pose_to_SE2,
    propagate_pose_rk4,
    riekf_A_matrix,
    riekf_landmark_H_matrix,
    riekf_Nhat_landmarks,
    riekf_Qhat_matrix,
    riekf_right_landmark_residual,
    se2_exp,
    wrap_angle,
)
from integrator import covariance_euler


class ContinuousDiscreteCarRIEKF:
    """Right-invariant EKF for landmark/range-bearing outputs."""

    def __init__(self, z0, P0, Q, N_each, landmarks, dt: float) -> None:
        self.z = np.asarray(z0, dtype=float).reshape(3).copy()
        self.z[0] = wrap_angle(self.z[0])
        self.P = np.asarray(P0, dtype=float).reshape(3, 3).copy()
        self.Q = np.asarray(Q, dtype=float).reshape(3, 3)
        self.N_each = np.asarray(N_each, dtype=float).reshape(2, 2)
        self.landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
        if len(self.landmarks) < 2:
            raise ValueError("RIEKF landmark output needs at least two distinct landmarks.")
        self.dt = float(dt)
        self.H = riekf_landmark_H_matrix(self.landmarks)

    def predict(self, u) -> None:
        # State propagation is still chi_dot = chi nu.
        self.z = propagate_pose_rk4(self.z, u, self.dt)

        # For the RIEKF landmark output in the paper, A_t = 0 and the process
        # noise covariance is transformed by the estimate-dependent matrix M.
        A = riekf_A_matrix()
        Qhat = riekf_Qhat_matrix(self.z, self.Q)
        self.P = covariance_euler(self.P, A, Qhat, self.dt)
        self.P += 1e-15 * np.eye(3)

    def update(self, y_landmarks) -> None:
        y_landmarks = np.asarray(y_landmarks, dtype=float).reshape(-1)
        residual = riekf_right_landmark_residual(self.z, y_landmarks, self.landmarks)
        Nhat = riekf_Nhat_landmarks(self.z, self.N_each, self.landmarks)

        S = self.H @ self.P @ self.H.T + Nhat
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T

        # RIEKF update is left multiplication: chi^+ = Exp(delta) chi.
        delta = K @ residual
        chi_plus = se2_exp(delta) @ pose_to_SE2(self.z)
        self.z = SE2_to_pose(chi_plus)

        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def step(self, u, y_landmarks=None) -> np.ndarray:
        self.predict(u)
        if y_landmarks is not None:
            self.update(y_landmarks)
        return self.z.copy()
