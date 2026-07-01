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
import jax.numpy as jnp


def wrap_angle(angle):
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_jax(angle):
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def rot2(theta: float) -> np.ndarray:
    """Planar rotation R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def unicycle_dynamics(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous unicycle dynamics z=[theta, x, y], u=[v, omega]."""
    theta = float(z[0])
    v, omega = float(u[0]), float(u[1])
    return np.array([omega, np.cos(theta) * v, np.sin(theta) * v], dtype=float)


def unicycle_dynamics_jax(z, u) -> jnp.ndarray:
    """Continuous unicycle dynamics z=[theta, x, y], u=[v, omega]."""
    theta = z[0]
    v, omega = u[0], u[1]
    return jnp.array([omega, jnp.cos(theta) * v, jnp.sin(theta) * v], dtype=float)


def ekf_A_matrix(theta_hat: float, v: float) -> np.ndarray:
    """Standard EKF linearized error matrix F_t from the paper's Sec. IV-D.

    Error convention: e = [theta_true - theta_hat, x_true - x_hat, y_true - y_hat].
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [-np.sin(theta_hat) * v, 0.0, 0.0],
            [np.cos(theta_hat) * v, 0.0, 0.0],
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
        A = 1.0 - alpha ** 2 / 6.0 + alpha ** 4 / 120.0
        B = alpha / 2.0 - alpha ** 3 / 24.0 + alpha ** 5 / 720.0
    else:
        A = np.sin(alpha) / alpha
        B = (1.0 - np.cos(alpha)) / alpha

    V = np.array([[A, -B], [B, A]], dtype=float)
    t = V @ rho
    chi = np.eye(3)
    chi[:2, :2] = R
    chi[:2, 2] = t
    return chi


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
        A = 1.0 - alpha ** 2 / 6.0 + alpha ** 4 / 120.0
        B = alpha / 2.0 - alpha ** 3 / 24.0 + alpha ** 5 / 720.0
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


# ============================================================
# Direct full non-convex LIEKF conditioning helpers on SE(2)
# ============================================================

def liekf_gps_innovation_SE2(z_pred: np.ndarray, y_gps: np.ndarray) -> np.ndarray:
    """Return z = Pi(chi_pred^{-1} Y - d) for the GPS output.

    With chi_pred=[R_pred,p_pred;0,1], Y=[y_gps;1], and
    d=[0,0,1]^T, the physical part of the invariant innovation is

        z = R_pred^T (y_gps - p_pred).
    """
    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    y_gps = np.asarray(y_gps, dtype=float).reshape(2)
    R_pred = rot2(z_pred[0])
    return R_pred.T @ (y_gps - z_pred[1:3])


def liekf_gps_covariance_SE2(z_pred: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Return N_hat=R_pred^T N R_pred for the invariant GPS innovation."""
    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    N = np.asarray(N, dtype=float).reshape(2, 2)
    R_pred = rot2(z_pred[0])
    N_hat = R_pred.T @ N @ R_pred
    return 0.5 * (N_hat + N_hat.T)


def liekf_gps_model_from_delta_SE2(delta: np.ndarray) -> np.ndarray:
    """Return phi(delta)=Pi((Exp(delta^)-I)d), d=[0,0,1]^T.

    For delta=[alpha,rho_1,rho_2], this is the translation block
    V(alpha)[rho_1,rho_2]^T of Exp(delta^), so it retains the exact
    finite-angle SE(2) nonlinearity.
    """
    return se2_exp(np.asarray(delta, dtype=float).reshape(3))[:2, 2].copy()


def liekf_gps_model_at_state_SE2(z_pred: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate phi(delta) for delta=Log(chi_pred^{-1}chi(z))."""
    return liekf_gps_model_from_delta_SE2(inv_retract_SE2(z_pred, z))


def liekf_gps_model_jacobian_SE2(
    z_pred: np.ndarray,
    z_iter: np.ndarray,
) -> np.ndarray:
    """Return d/deta phi(delta(chi_iter Exp(eta^))) at eta=0."""
    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    z_iter = np.asarray(z_iter, dtype=float).reshape(3)

    def model_pert(eta):
        z_candidate = retract_SE2(z_iter, eta)
        return liekf_gps_model_at_state_SE2(z_pred, z_candidate)

    return numerical_jacobian_zero(model_pert, 3)


# ============================================================
# Full homogeneous-vector / full-matrix conditioning utilities
# ============================================================

SE2_GPS_D = np.array([0.0, 0.0, 1.0], dtype=float)


def as_full_homogeneous_measurement_SE2(
    Y_or_y: np.ndarray,
) -> np.ndarray:
    """Return a full 3-vector observation for the SE(2) action.

    Parameters
    ----------
    Y_or_y:
        Either a physical GPS observation [y_x, y_y] or an already embedded
        homogeneous vector Y=[y_x, y_y, 1].

    Notes
    -----
    The 2-vector case is retained only for backward compatibility with the
    existing filter API.  All conditioning expressions subsequently use the
    full homogeneous vector.
    """
    arr = np.asarray(Y_or_y, dtype=float).reshape(-1)
    if arr.size == 2:
        return np.array([arr[0], arr[1], 1.0], dtype=float)
    if arr.size == 3:
        return arr.copy()
    raise ValueError(
        "Y_or_y must have length 2 (physical GPS) or 3 (full homogeneous observation)."
    )


def as_full_homogeneous_covariance_SE2(
    N_or_Nxy: np.ndarray,
    homogeneous_variance: float = 1.0,
) -> np.ndarray:
    """Return an SPD 3x3 covariance for a full homogeneous observation.

    A physical GPS covariance is 2x2 because the last homogeneous coordinate
    is exactly one and has no physical measurement noise.  A literal full
    Mahalanobis norm requires an invertible covariance, so for a 2x2 input we
    embed it as diag(N_xy, homogeneous_variance).

    This dummy third variance does *not* change the exact conditioning
    objective: the third component of

        X_pred^{-1} Y - d - (Exp(delta^) - I)d

    is identically zero for the standard SE(2) GPS observation d=[0,0,1].
    It simply makes the full-matrix information-form identity well-defined.

    A supplied 3x3 covariance is used directly and must be SPD.
    """
    N = np.asarray(N_or_Nxy, dtype=float)
    if N.shape == (2, 2):
        if homogeneous_variance <= 0.0:
            raise ValueError("homogeneous_variance must be positive.")
        Nfull = np.zeros((3, 3), dtype=float)
        Nfull[:2, :2] = N
        Nfull[2, 2] = float(homogeneous_variance)
    elif N.shape == (3, 3):
        Nfull = N.copy()
    else:
        raise ValueError("N must have shape (2,2) or (3,3).")

    Nfull = 0.5 * (Nfull + Nfull.T)
    vals = np.linalg.eigvalsh(Nfull)
    if np.min(vals) <= 0.0:
        raise ValueError(
            "The full observation covariance must be SPD. For physical 2D GPS, "
            "pass a SPD 2x2 covariance and this function will add the dummy "
            "homogeneous variance automatically."
        )
    return Nfull


def full_left_invariant_innovation_SE2(
    chi_pred: np.ndarray,
    Y: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Compute z = chi_pred^{-1} Y - d using full 3-vectors."""
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    Y = np.asarray(Y, dtype=float).reshape(3)
    d = np.asarray(d, dtype=float).reshape(3)
    return np.linalg.solve(chi_pred, Y) - d


def full_left_invariant_covariance_SE2(
    chi_pred: np.ndarray,
    N_full: np.ndarray,
) -> np.ndarray:
    """Compute N_hat = chi_pred^{-1} N chi_pred^{-T}."""
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    N_full = np.asarray(N_full, dtype=float).reshape(3, 3)
    chi_pred_inv = np.linalg.inv(chi_pred)
    N_hat = chi_pred_inv @ N_full @ chi_pred_inv.T
    return 0.5 * (N_hat + N_hat.T)


def full_left_invariant_information_SE2(
    chi_pred: np.ndarray,
    N_full: np.ndarray,
) -> np.ndarray:
    """Compute N_hat^{-1}=chi_pred^T N^{-1} chi_pred.

    This is exactly the full-matrix identity requested in the derivation.  The
    argument ``chi_pred`` is the predicted *group matrix*, not a 3-vector pose.
    """
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    N_full = np.asarray(N_full, dtype=float).reshape(3, 3)
    Ninv = np.linalg.inv(N_full)
    Nhatinv = chi_pred.T @ Ninv @ chi_pred
    return 0.5 * (Nhatinv + Nhatinv.T)


def full_prior_residual_SE2(
    chi_pred: np.ndarray,
    chi: np.ndarray,
) -> np.ndarray:
    """Return delta=Log(chi_pred^{-1} chi) in se(2) coordinates."""
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    chi = np.asarray(chi, dtype=float).reshape(3, 3)
    return se2_log(np.linalg.solve(chi_pred, chi))


def full_prior_residual_jacobian_SE2(
    chi_pred: np.ndarray,
    chi_iter: np.ndarray,
) -> np.ndarray:
    """d/deta Log(chi_pred^{-1} chi_iter Exp(eta^)) evaluated at eta=0."""
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    chi_iter = np.asarray(chi_iter, dtype=float).reshape(3, 3)

    def prior_pert(eta):
        return full_prior_residual_SE2(chi_pred, chi_iter @ se2_exp(eta))

    return numerical_jacobian_zero(prior_pert, 3)


def full_left_invariant_measurement_model_SE2(
    chi_pred: np.ndarray,
    chi: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Compute phi = (chi_pred^{-1} chi - I) d using full states.

    If chi=chi_pred Exp(delta^), this equals

        phi(delta) = (Exp(delta^) - I) d.

    It is the exact nonlinear quantity appearing in the invariant
    conditioning objective; no rotation block or reduced output is extracted.
    """
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    chi = np.asarray(chi, dtype=float).reshape(3, 3)
    d = np.asarray(d, dtype=float).reshape(3)
    relative = np.linalg.solve(chi_pred, chi)
    return (relative - np.eye(3)) @ d


def full_left_invariant_measurement_jacobian_SE2(
    chi_pred: np.ndarray,
    chi_iter: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """d/deta [(chi_pred^{-1} chi_iter Exp(eta^) - I)d] at eta=0."""
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    chi_iter = np.asarray(chi_iter, dtype=float).reshape(3, 3)
    d = np.asarray(d, dtype=float).reshape(3)

    def model_pert(eta):
        chi_candidate = chi_iter @ se2_exp(eta)
        return full_left_invariant_measurement_model_SE2(
            chi_pred, chi_candidate, d
        )

    return numerical_jacobian_zero(model_pert, 3)


def full_world_position_jacobian_SE2(chi_iter: np.ndarray) -> np.ndarray:
    """d/deta position(chi_iter Exp(eta^)) at eta=0 in world coordinates."""
    chi_iter = np.asarray(chi_iter, dtype=float).reshape(3, 3)

    def position_pert(eta):
        return (chi_iter @ se2_exp(eta))[:2, 2]

    return numerical_jacobian_zero(position_pert, 3)


# ============================================================
# Fixed predicted-chart utilities for SCP in delta coordinates
# ============================================================

def full_left_invariant_measurement_model_from_delta_SE2(
    delta: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Return phi(delta) = (Exp(delta^) - I) d.

    This is the exact full homogeneous measurement map in the fixed coordinate
    chart chi = chi_pred Exp(delta^).  It has no dependence on chi_pred.
    """
    delta = np.asarray(delta, dtype=float).reshape(3)
    d = np.asarray(d, dtype=float).reshape(3)
    return (se2_exp(delta) - np.eye(3)) @ d


def full_left_invariant_measurement_jacobian_from_delta_SE2(
    delta: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Return d phi(delta) / d delta in the fixed predicted chart.

    For an additive SCP step s in the Lie-algebra coordinate,

        phi(delta + s) ~= phi(delta) + H_delta s.

    The derivative is evaluated numerically to remain general and exactly
    consistent with the SE(2) exponential used by the nonlinear objective.
    """
    delta = np.asarray(delta, dtype=float).reshape(3)
    d = np.asarray(d, dtype=float).reshape(3)

    def model_pert(step):
        return full_left_invariant_measurement_model_from_delta_SE2(
            delta + np.asarray(step, dtype=float).reshape(3), d
        )

    return numerical_jacobian_zero(model_pert, 3)


def full_world_position_from_delta_SE2(
    chi_pred: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """World-frame position of chi_pred Exp(delta^) without storing chi_i."""
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    delta = np.asarray(delta, dtype=float).reshape(3)
    return (chi_pred @ se2_exp(delta))[:2, 2].copy()


def full_world_position_jacobian_from_delta_SE2(
    chi_pred: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Return d p(chi_pred Exp(delta^)) / d delta.

    This is the Jacobian needed for the physical world-frame obstacle
    constraints when the SCP decision variable is the fixed-chart increment s.
    """
    chi_pred = np.asarray(chi_pred, dtype=float).reshape(3, 3)
    delta = np.asarray(delta, dtype=float).reshape(3)

    def position_pert(step):
        return full_world_position_from_delta_SE2(
            chi_pred, delta + np.asarray(step, dtype=float).reshape(3)
        )

    return numerical_jacobian_zero(position_pert, 3)
