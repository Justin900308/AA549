"""Continuous-discrete additive quaternion EKF.

The propagation is continuous-time quaternion dynamics integrated every dt.
The conditioning/update is discrete and can be called every update_stride*dt.
"""

from __future__ import annotations

import numpy as np

from dynamics import (
    gyro_input_matrix,
    measurement_model_left,
    normalize_quat,
    quat_dynamics,
    quat_state_jacobian,
)
from integrator import RK4
from linearization_lib import measurement_jacobian_quat


class ContinuousDiscreteQuaternionEKF:
    """Additive EKF on the ambient quaternion coordinates q in R^4."""

    def __init__(
        self,
        q0,
        P0,
        Q_gyro,
        R_meas,
        reference_body_vectors,
        dt: float,
    ) -> None:
        self.q = normalize_quat(q0)
        self.P = np.asarray(P0, dtype=float).reshape(4, 4)
        self.Q_gyro = np.asarray(Q_gyro, dtype=float).reshape(3, 3)
        self.R_meas = np.asarray(R_meas, dtype=float)
        self.reference_body_vectors = np.asarray(reference_body_vectors, dtype=float).reshape(-1, 3)
        self.dt = float(dt)

    def predict(self, omega_meas) -> None:
        """Continuous prediction over one dt interval."""
        omega_meas = np.asarray(omega_meas, dtype=float).reshape(3)
        q_old = self.q.copy()

        # State propagation.
        self.q = RK4(self.q, omega_meas, quat_dynamics, self.dt)

        # Continuous-discrete covariance propagation: Pdot = A P + P A^T + G Q G^T.
        A = quat_state_jacobian(omega_meas)
        G = 0.5 * gyro_input_matrix(q_old)
        Pdot = A @ self.P + self.P @ A.T + G @ self.Q_gyro @ G.T
        self.P = self.P + self.dt * Pdot
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(4)

    def update(self, y) -> None:
        """Discrete conditioning step using stacked vector measurements."""
        y = np.asarray(y, dtype=float).reshape(-1)
        h = measurement_model_left(self.q, self.reference_body_vectors)
        residual = y - h
        H = measurement_jacobian_quat(self.q, self.reference_body_vectors)

        S = H @ self.P @ H.T + self.R_meas
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        dq = K @ residual
        self.q = normalize_quat(self.q + dq)

        # Joseph form is safer numerically than (I-KH)P.
        I = np.eye(4)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(4)

    def step(self, omega_meas, y=None) -> np.ndarray:
        self.predict(omega_meas)
        if y is not None:
            self.update(y)
        return self.q.copy()
