"""Dynamics and SE(2) utilities for the simplified-car EKF/LIEKF example.

This file follows the notation of Barrau and Bonnabel, TAC 2017, Sec. IV:

    theta_dot = omega
    x_dot     = cos(theta) v
    y_dot     = sin(theta) v

and the SE(2) embedding

    chi = [[R(theta), x],
           [0, 0,       1]].
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot2(theta: float) -> np.ndarray:
    """Planar rotation R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def unicycle_dynamics(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous unicycle dynamics z=[theta, x, y], u=[v, omega]."""
    theta = float(z[0])
    v, omega = float(u[0]), float(u[1])
    return np.array([omega, np.cos(theta) * v, np.sin(theta) * v], dtype=float)


def ekf_A_matrix(theta_hat: float, v: float) -> np.ndarray:
    """Standard EKF linearized error matrix F_t from the paper's Sec. IV-D.

    Error convention: e = [theta_true - theta_hat, x_true - x_hat, y_true - y_hat].
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [-np.sin(theta_hat) * v, 0.0, 0.0],
            [ np.cos(theta_hat) * v, 0.0, 0.0],
        ],
        dtype=float,
    )


def liekf_A_matrix(v: float, omega: float) -> np.ndarray:
    """LIEKF log-error propagation matrix A_t = -ad_mu from Sec. IV-B3.

    mu = [omega, v, 0].  The paper writes

        xi_dot = - [[0, 0, 0], [0, 0, -omega], [-v, omega, 0]] xi - beta.
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, omega],
            [v, -omega, 0.0],
        ],
        dtype=float,
    )


def gps_H_matrix() -> np.ndarray:
    """GPS position observation matrix H = [0_{2,1}, I_2]."""
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def pose_to_SE2(z: np.ndarray) -> np.ndarray:
    """Map z=[theta,x,y] to chi in SE(2)."""
    theta, x, y = float(z[0]), float(z[1]), float(z[2])
    chi = np.eye(3)
    chi[:2, :2] = rot2(theta)
    chi[:2, 2] = np.array([x, y])
    return chi


def SE2_to_pose(chi: np.ndarray) -> np.ndarray:
    """Map chi in SE(2) back to z=[theta,x,y]."""
    theta = np.arctan2(chi[1, 0], chi[0, 0])
    return np.array([wrap_angle(theta), chi[0, 2], chi[1, 2]], dtype=float)


def se2_wedge(xi: np.ndarray) -> np.ndarray:
    """Wedge map L_se(2)(xi), xi=[alpha,u1,u2]."""
    alpha, u1, u2 = np.asarray(xi, dtype=float).reshape(3)
    return np.array(
        [[0.0, -alpha, u1], [alpha, 0.0, u2], [0.0, 0.0, 0.0]],
        dtype=float,
    )


def se2_exp(xi: np.ndarray) -> np.ndarray:
    """Closed-form SE(2) exponential for xi=[alpha,u1,u2]."""
    alpha, u1, u2 = np.asarray(xi, dtype=float).reshape(3)
    R = rot2(alpha)
    rho = np.array([u1, u2], dtype=float)

    if abs(alpha) < 1e-10:
        # V = I + alpha/2 J + alpha^2/6 J^2 + ...
        A = 1.0 - alpha**2 / 6.0 + alpha**4 / 120.0
        B = alpha / 2.0 - alpha**3 / 24.0 + alpha**5 / 720.0
    else:
        A = np.sin(alpha) / alpha
        B = (1.0 - np.cos(alpha)) / alpha

    V = np.array([[A, -B], [B, A]], dtype=float)
    t = V @ rho
    chi = np.eye(3)
    chi[:2, :2] = R
    chi[:2, 2] = t
    return chi


def propagate_pose_rk4(z: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step for the unicycle state."""
    z = np.asarray(z, dtype=float).reshape(3)
    u = np.asarray(u, dtype=float).reshape(2)
    k1 = unicycle_dynamics(z, u)
    k2 = unicycle_dynamics(z + 0.5 * dt * k1, u)
    k3 = unicycle_dynamics(z + 0.5 * dt * k2, u)
    k4 = unicycle_dynamics(z + dt * k3, u)
    zp1 = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    zp1[0] = wrap_angle(zp1[0])
    return zp1


def liekf_left_gps_residual(z_hat: np.ndarray, y_gps: np.ndarray) -> np.ndarray:
    """Reduced LIEKF residual ptilde(chi_hat^{-1}Y-d).

    For GPS Y=[x_true+V;1] and d=[0,0,1], this is

        r = R(theta_hat)^T (y_gps - x_hat).
    """
    theta_hat = float(z_hat[0])
    x_hat = np.asarray(z_hat[1:3], dtype=float)
    y_gps = np.asarray(y_gps, dtype=float).reshape(2)
    return rot2(theta_hat).T @ (y_gps - x_hat)


def heading_error_deg(z_true: np.ndarray, z_hat: np.ndarray) -> float:
    return float(abs(np.rad2deg(wrap_angle(z_true[0] - z_hat[0]))))


def position_error(z_true: np.ndarray, z_hat: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(z_true[1:3]) - np.asarray(z_hat[1:3])))


# ============================================================
# Extra intrinsic/SCP utilities used by continuous_discrete_iscvx.py
# ============================================================

def se2_log(chi: np.ndarray) -> np.ndarray:
    """Closed-form SE(2) logarithm, inverse of se2_exp.

    Returns xi=[alpha,u1,u2] such that se2_exp(xi) = chi.
    """
    chi = np.asarray(chi, dtype=float).reshape(3, 3)
    alpha = wrap_angle(np.arctan2(chi[1, 0], chi[0, 0]))
    t = chi[:2, 2]

    if abs(alpha) < 1e-10:
        A = 1.0 - alpha**2 / 6.0 + alpha**4 / 120.0
        B = alpha / 2.0 - alpha**3 / 24.0 + alpha**5 / 720.0
    else:
        A = np.sin(alpha) / alpha
        B = (1.0 - np.cos(alpha)) / alpha

    V = np.array([[A, -B], [B, A]], dtype=float)
    rho = np.linalg.solve(V, t)
    return np.array([alpha, rho[0], rho[1]], dtype=float)


def retract_SE2(z_base: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Right retraction R_z(eta) = chi(z) Exp(eta), returned as z=[theta,x,y]."""
    chi = pose_to_SE2(z_base) @ se2_exp(eta)
    return SE2_to_pose(chi)


def inv_retract_SE2(z_base: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Inverse right retraction Log(chi_base^{-1} chi(z))."""
    chi_rel = np.linalg.inv(pose_to_SE2(z_base)) @ pose_to_SE2(z)
    return se2_log(chi_rel)


def gps_measurement_model(z: np.ndarray) -> np.ndarray:
    """GPS measurement h(z) = position = [x,y]."""
    z = np.asarray(z, dtype=float).reshape(3)
    return z[1:3].copy()


def numerical_jacobian_zero(fun, xdim: int, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Jacobian of fun(e) at e=0."""
    f0 = np.asarray(fun(np.zeros(xdim)), dtype=float).reshape(-1)
    J = np.zeros((f0.size, xdim), dtype=float)
    for i in range(xdim):
        e = np.zeros(xdim)
        e[i] = eps
        fp = np.asarray(fun(e), dtype=float).reshape(-1)
        fm = np.asarray(fun(-e), dtype=float).reshape(-1)
        J[:, i] = (fp - fm) / (2.0 * eps)
    return J


def prior_residual_jacobian_SE2(z_pred: np.ndarray, z_iter: np.ndarray) -> np.ndarray:
    """Jp = d/deta Log(chi_pred^{-1} chi_iter Exp(eta)) at eta=0."""
    def prior_pert(eta):
        return inv_retract_SE2(z_pred, retract_SE2(z_iter, eta))

    return numerical_jacobian_zero(prior_pert, 3)


def gps_measurement_jacobian_intrinsic(z_iter: np.ndarray) -> np.ndarray:
    """H = d/deta h(chi_iter Exp(eta)) at eta=0."""
    def meas_pert(eta):
        return gps_measurement_model(retract_SE2(z_iter, eta))

    return numerical_jacobian_zero(meas_pert, 3)
