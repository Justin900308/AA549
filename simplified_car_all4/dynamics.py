"""Dynamics and SE(2) utilities for the simplified-car EKF/LIEKF/RIEKF/ISCVX comparison.

The simplified-car dynamics follow Barrau and Bonnabel, TAC 2017, Sec. IV:

    theta_dot = omega
    x_dot     = cos(theta) v
    y_dot     = sin(theta) v

State convention throughout the code:

    z = [theta, x, y]

and the SE(2) embedding is

    chi = [[R(theta), p],
           [0, 0,       1]].
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def wrap_angle(angle: float | Array) -> float | Array:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot2(theta: float) -> Array:
    """Planar rotation R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def unicycle_dynamics(z: Array, u: Array) -> Array:
    """Continuous unicycle dynamics z=[theta,x,y], u=[v,omega]."""
    theta = float(z[0])
    v, omega = float(u[0]), float(u[1])
    return np.array([omega, np.cos(theta) * v, np.sin(theta) * v], dtype=float)


def propagate_pose_rk4(z: Array, u: Array, dt: float) -> Array:
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


# -----------------------------------------------------------------------------
# Linearized propagation matrices.
# -----------------------------------------------------------------------------


def ekf_A_matrix(theta_hat: float, v: float) -> Array:
    """Standard EKF linearized error matrix F_t from the paper's Sec. IV-D.

    Error convention: e = [theta_true-theta_hat, x_true-x_hat, y_true-y_hat].
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [-np.sin(theta_hat) * v, 0.0, 0.0],
            [ np.cos(theta_hat) * v, 0.0, 0.0],
        ],
        dtype=float,
    )


def liekf_A_matrix(v: float, omega: float) -> Array:
    """LIEKF log-error propagation matrix A_t=-ad_mu from Sec. IV-B3.

    The paper has mu=[omega,v,0] and

        xi_dot = -[[0,0,0],[0,0,-omega],[-v,omega,0]] xi - beta.
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, omega],
            [v, -omega, 0.0],
        ],
        dtype=float,
    )


def riekf_A_matrix() -> Array:
    """RIEKF propagation matrix for the landmark/right-invariant output case.

    In Sec. IV-B4 of the paper, A_t = 0_{3,3}.
    """
    return np.zeros((3, 3), dtype=float)


def gps_H_matrix() -> Array:
    """GPS position observation matrix H=[0_{2,1}, I_2]."""
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def riekf_landmark_H_matrix(landmarks: Array) -> Array:
    """RIEKF landmark observation matrix from Sec. IV-B4.

    For each landmark p_k=(p_x,p_y), the two rows are

        [-p_y, 1, 0]
        [ p_x, 0, 1].
    """
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    rows = []
    for px, py in landmarks:
        rows.append([-py, 1.0, 0.0])
        rows.append([ px, 0.0, 1.0])
    return np.asarray(rows, dtype=float)


def riekf_Qhat_matrix(z_hat: Array, Q: Array) -> Array:
    """RIEKF transformed propagation covariance Qhat from Sec. IV-B4.

    Qhat = M(z_hat) Q M(z_hat)^T, with

        M = [[1, 0, 0],
             [x_y, R11, R12],
             [-x_x, R21, R22]].
    """
    z_hat = np.asarray(z_hat, dtype=float).reshape(3)
    theta, x, y = float(z_hat[0]), float(z_hat[1]), float(z_hat[2])
    R = rot2(theta)
    M = np.array(
        [
            [1.0, 0.0, 0.0],
            [y, R[0, 0], R[0, 1]],
            [-x, R[1, 0], R[1, 1]],
        ],
        dtype=float,
    )
    return M @ np.asarray(Q, dtype=float).reshape(3, 3) @ M.T


# -----------------------------------------------------------------------------
# SE(2) maps and right retraction.
# -----------------------------------------------------------------------------


def pose_to_SE2(z: Array) -> Array:
    """Map z=[theta,x,y] to chi in SE(2)."""
    theta, x, y = float(z[0]), float(z[1]), float(z[2])
    chi = np.eye(3)
    chi[:2, :2] = rot2(theta)
    chi[:2, 2] = np.array([x, y])
    return chi


def SE2_to_pose(chi: Array) -> Array:
    """Map chi in SE(2) back to z=[theta,x,y]."""
    chi = np.asarray(chi, dtype=float).reshape(3, 3)
    theta = np.arctan2(chi[1, 0], chi[0, 0])
    return np.array([wrap_angle(theta), chi[0, 2], chi[1, 2]], dtype=float)


def se2_wedge(xi: Array) -> Array:
    """Wedge map L_se(2)(xi), xi=[alpha,u1,u2]."""
    alpha, u1, u2 = np.asarray(xi, dtype=float).reshape(3)
    return np.array([[0.0, -alpha, u1], [alpha, 0.0, u2], [0.0, 0.0, 0.0]], dtype=float)


def se2_exp(xi: Array) -> Array:
    """Closed-form SE(2) exponential for xi=[alpha,u1,u2]."""
    alpha, u1, u2 = np.asarray(xi, dtype=float).reshape(3)
    R = rot2(alpha)
    rho = np.array([u1, u2], dtype=float)

    if abs(alpha) < 1e-10:
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


def se2_log(chi: Array) -> Array:
    """Closed-form SE(2) logarithm, inverse of se2_exp."""
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


def retract_SE2(z_base: Array, eta: Array) -> Array:
    """Right retraction R_z(eta)=chi(z) Exp(eta), returned as z=[theta,x,y]."""
    chi = pose_to_SE2(z_base) @ se2_exp(eta)
    return SE2_to_pose(chi)


def inv_retract_SE2(z_base: Array, z: Array) -> Array:
    """Inverse right retraction Log(chi_base^{-1} chi(z))."""
    chi_rel = np.linalg.inv(pose_to_SE2(z_base)) @ pose_to_SE2(z)
    return se2_log(chi_rel)


# -----------------------------------------------------------------------------
# Measurements and residuals.
# -----------------------------------------------------------------------------


def gps_measurement_model(z: Array) -> Array:
    """GPS measurement h(z)=position=[x,y]."""
    z = np.asarray(z, dtype=float).reshape(3)
    return z[1:3].copy()


def liekf_left_gps_residual(z_hat: Array, y_gps: Array) -> Array:
    """Reduced LIEKF residual ptilde(chi_hat^{-1}Y-d).

    For GPS Y=[x_true+V;1] and d=[0,0,1], this is

        r = R(theta_hat)^T (y_gps - x_hat).
    """
    theta_hat = float(z_hat[0])
    x_hat = np.asarray(z_hat[1:3], dtype=float)
    y_gps = np.asarray(y_gps, dtype=float).reshape(2)
    return rot2(theta_hat).T @ (y_gps - x_hat)


def landmark_body_measurement_model(z: Array, landmarks: Array) -> Array:
    """Right-invariant landmark/range-bearing output from paper Eq. (40)/(46).

    For landmark p_k, the body-frame relative observation is

        y_k = R(theta)^T (x - p_k).

    The homogeneous vector used by RIEKF is [y_k; -1], because then
    chi [y_k; -1] + [p_k; 1] = 0 at the true state.
    """
    z = np.asarray(z, dtype=float).reshape(3)
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    R_T = rot2(float(z[0])).T
    x = z[1:3]
    return np.concatenate([R_T @ (x - p) for p in landmarks])


def riekf_right_landmark_residual(z_hat: Array, y_landmarks: Array, landmarks: Array) -> Array:
    """Reduced RIEKF residual for right-invariant landmark output.

    For each landmark p_k and measurement y_k=R_true^T(x_true-p_k),

        r_k = R_hat y_k - x_hat + p_k.

    At the true state and zero noise this residual is zero.  Its first-order
    dependence on the right-invariant error is r ≈ -H xi + noise.
    """
    z_hat = np.asarray(z_hat, dtype=float).reshape(3)
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    y_landmarks = np.asarray(y_landmarks, dtype=float).reshape(-1, 2)
    Rhat = rot2(float(z_hat[0]))
    xhat = z_hat[1:3]
    return np.concatenate([Rhat @ yk - xhat + pk for yk, pk in zip(y_landmarks, landmarks)])


def block_diag_2x2(blocks: list[Array]) -> Array:
    """Minimal block diagonal helper for 2x2 covariance blocks."""
    n = len(blocks)
    out = np.zeros((2 * n, 2 * n), dtype=float)
    for i, B in enumerate(blocks):
        out[2*i:2*i+2, 2*i:2*i+2] = B
    return out


def riekf_Nhat_landmarks(z_hat: Array, N_each: Array, landmarks: Array) -> Array:
    """RIEKF transformed landmark measurement covariance.

    Nhat is block diagonal with blocks R(theta_hat) N_each R(theta_hat)^T.
    """
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    Rhat = rot2(float(np.asarray(z_hat).reshape(3)[0]))
    block = Rhat @ np.asarray(N_each, dtype=float).reshape(2, 2) @ Rhat.T
    return block_diag_2x2([block for _ in range(len(landmarks))])


# -----------------------------------------------------------------------------
# Numerical Jacobians for ISCVX.
# -----------------------------------------------------------------------------


def numerical_jacobian_zero(fun, xdim: int, eps: float = 1e-6) -> Array:
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


def prior_residual_jacobian_SE2(z_pred: Array, z_iter: Array) -> Array:
    """Jp=d/deta Log(chi_pred^{-1} chi_iter Exp(eta)) at eta=0."""
    def prior_pert(eta):
        return inv_retract_SE2(z_pred, retract_SE2(z_iter, eta))

    return numerical_jacobian_zero(prior_pert, 3)


def gps_measurement_jacobian_intrinsic(z_iter: Array) -> Array:
    """H=d/deta h(chi_iter Exp(eta)) at eta=0."""
    def meas_pert(eta):
        return gps_measurement_model(retract_SE2(z_iter, eta))

    return numerical_jacobian_zero(meas_pert, 3)


# -----------------------------------------------------------------------------
# Error metrics.
# -----------------------------------------------------------------------------


def heading_error_deg(z_true: Array, z_hat: Array) -> float:
    return float(abs(np.rad2deg(wrap_angle(z_true[0] - z_hat[0]))))


def position_error(z_true: Array, z_hat: Array) -> float:
    return float(np.linalg.norm(np.asarray(z_true[1:3]) - np.asarray(z_hat[1:3])))
