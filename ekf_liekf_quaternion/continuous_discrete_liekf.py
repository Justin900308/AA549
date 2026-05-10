"""Continuous-discrete Left-Invariant EKF (LIEKF) for quaternion attitude.

State: q in S^3, representing R(q) from body frame to inertial frame.
Error: eta = R_true^T R_hat = Exp(xi), xi in R^3.

For dynamics R_dot = R [omega]x, the left-invariant log-error propagation is

    xi_dot = -[omega]x xi - gyro_noise,

so the covariance is only 3x3.  The update used here is the left-invariant
observation model y_i = R(q_true) d_i + noise, with residual

    r_i = R(q_hat)^T y_i - d_i = [d_i]x xi + noise + higher-order terms.

The correction is applied intrinsically:

    q_hat^+ = q_hat * Exp(delta),       delta = -K r.
"""

from __future__ import annotations

import numpy as np

from dynamics import liekf_left_residual, normalize_quat, quat_dynamics, quat_exp, quat_mul, skew
from integrator import RK4
from linearization_lib import measurement_jacobian_liekf


class ContinuousDiscreteQuaternionLIEKF:
    """Left-invariant EKF with 3D tangent-space covariance."""

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
        self.P = np.asarray(P0, dtype=float).reshape(3, 3)
        self.Q_gyro = np.asarray(Q_gyro, dtype=float).reshape(3, 3)
        self.R_meas = np.asarray(R_meas, dtype=float)
        self.reference_body_vectors = np.asarray(reference_body_vectors, dtype=float).reshape(-1, 3)
        self.dt = float(dt)

    def predict(self, omega_meas) -> None:
        """Continuous prediction over one dt interval."""
        omega_meas = np.asarray(omega_meas, dtype=float).reshape(3)
        self.q = RK4(self.q, omega_meas, quat_dynamics, self.dt)

        A = -skew(omega_meas)
        Pdot = A @ self.P + self.P @ A.T + self.Q_gyro
        self.P = self.P + self.dt * Pdot
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(3)

    def update(self, y) -> None:
        """Discrete conditioning step in the tangent space of SO(3)."""
        residual = liekf_left_residual(self.q, y, self.reference_body_vectors)
        H = measurement_jacobian_liekf(self.reference_body_vectors)

        S = H @ self.P @ H.T + self.R_meas
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        # Since residual = H xi + noise, q_hat^+ = q_hat Exp(-K residual).
        delta = -K @ residual
        self.q = normalize_quat(quat_mul(self.q, quat_exp(delta)))

        I = np.eye(3)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-12 * np.eye(3)

    def step(self, omega_meas, y=None) -> np.ndarray:
        self.predict(omega_meas)
        if y is not None:
            self.update(y)
        return self.q.copy()
