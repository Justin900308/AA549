"""Continuous-discrete intrinsic/SCP filter for the simplified car.

This class keeps the same continuous prediction as the LIEKF:

    z_pred = flow(z_hat, u),
    Pdot   = A_liekf P + P A_liekf^T + Q.

Only the discrete conditioning step is replaced by an intrinsic convexified
least-squares update, following the same pattern as test2.py:

    chi = chi_iter Exp(eta),

    min_eta || Log(chi_pred^{-1} chi_iter Exp(eta)) ||_{P_pred^{-1}}^2
          + || y_gps - h(chi_iter Exp(eta)) ||_{N^{-1}}^2
    s.t.  ||eta|| <= trust_radius,

where both residuals are linearized at the current iterate at each SCP step.
"""

from __future__ import annotations

import numpy as np
from dynamics import (
    gps_measurement_jacobian_intrinsic,
    gps_measurement_model,
    inv_retract_SE2,
    liekf_A_matrix,
    prior_residual_jacobian_SE2,
    propagate_pose_rk4,
    retract_SE2,
    wrap_angle,
)
from integrator import covariance_euler


class ContinuousDiscreteCarISCVX:
    """LIEKF prediction + intrinsic convexified GPS conditioning on SE(2)."""

    def __init__(
        self,
        z0,
        P0,
        Q,
        N,
        dt: float,
        trust_radius: float = 0.5,
        max_scp_iters: int = 5,
        tol: float = 1e-9,
    ) -> None:
        self.z = np.asarray(z0, dtype=float).reshape(3).copy()
        self.z[0] = wrap_angle(self.z[0])
        self.P = np.asarray(P0, dtype=float).reshape(3, 3).copy()
        self.Q = np.asarray(Q, dtype=float).reshape(3, 3)
        self.N = np.asarray(N, dtype=float).reshape(2, 2)
        self.dt = float(dt)
        self.trust_radius = float(trust_radius)
        self.max_scp_iters = int(max_scp_iters)
        self.tol = float(tol)

    def predict(self, u) -> None:
        """Same prediction step as the LIEKF."""
        u = np.asarray(u, dtype=float).reshape(2)
        v, omega = float(u[0]), float(u[1])
        A = liekf_A_matrix(v, omega)

        self.z = propagate_pose_rk4(self.z, u, self.dt)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def update(self, y_gps) -> None:
        self.z, self.P = intrinsic_cvx_update_SE2(
            z_pred=self.z,
            P_pred=self.P,
            y_gps=np.asarray(y_gps, dtype=float).reshape(2),
            N=self.N,
            trust_radius=self.trust_radius,
            max_scp_iters=self.max_scp_iters,
            tol=self.tol,
        )

    def step(self, u, y_gps=None) -> np.ndarray:
        self.predict(u)
        if y_gps is not None:
            self.update(y_gps)
        return self.z.copy()



def solve_trust_region_qp(
    Jp: np.ndarray,
    Pinv: np.ndarray,
    rp: np.ndarray,
    H: np.ndarray,
    Ninv: np.ndarray,
    ry: np.ndarray,
    trust_radius: float,
) -> np.ndarray:
    """Solve the small convex trust-region QP exactly.

    The SCP subproblem is

        min_eta ||rp + Jp eta||_Pinv^2 + ||ry - H eta||_Ninv^2
        s.t.    ||eta|| <= trust_radius.

    This is the same convex subproblem as the CVXPY version in test2.py, but
    for this 3D problem a direct trust-region solve avoids a runtime dependency
    on cvxpy.
    """
    Hess = Jp.T @ Pinv @ Jp + H.T @ Ninv @ H
    grad = Jp.T @ Pinv @ rp - H.T @ Ninv @ ry
    Hess = 0.5 * (Hess + Hess.T) + 1e-14 * np.eye(3)

    try:
        eta_unc = -np.linalg.solve(Hess, grad)
    except np.linalg.LinAlgError:
        eta_unc = -np.linalg.pinv(Hess) @ grad

    if np.linalg.norm(eta_unc) <= trust_radius:
        return eta_unc

    # Boundary solution: (Hess + lambda I) eta = -grad, lambda >= 0,
    # with ||eta|| = trust_radius.  The norm is monotone in lambda.
    lam_low = 0.0
    lam_high = 1.0
    I = np.eye(3)

    def eta_of_lam(lam):
        return -np.linalg.solve(Hess + lam * I, grad)

    while np.linalg.norm(eta_of_lam(lam_high)) > trust_radius:
        lam_high *= 2.0
        if lam_high > 1e14:
            break

    for _ in range(80):
        lam_mid = 0.5 * (lam_low + lam_high)
        eta_mid = eta_of_lam(lam_mid)
        if np.linalg.norm(eta_mid) > trust_radius:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    return eta_of_lam(lam_high)

def intrinsic_cvx_update_SE2(
    z_pred: np.ndarray,
    P_pred: np.ndarray,
    y_gps: np.ndarray,
    N: np.ndarray,
    trust_radius: float = 0.5,
    max_scp_iters: int = 5,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Successive convexified intrinsic GPS update on SE(2).

    This mirrors the quaternion intrinsic update in test2.py, but uses the
    SE(2) right retraction chi = chi_iter Exp(eta).
    """
    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    z_iter = z_pred.copy()
    P_pred = np.asarray(P_pred, dtype=float).reshape(3, 3)
    N = np.asarray(N, dtype=float).reshape(2, 2)

    # Small regularization protects the inverse when the initial position is
    # assumed exactly known and P has tiny translational entries.
    Pinv = np.linalg.inv(P_pred + 1e-12 * np.eye(3))
    Ninv = np.linalg.inv(N + 1e-12 * np.eye(2))
    Pinv = 0.5 * (Pinv + Pinv.T)
    Ninv = 0.5 * (Ninv + Ninv.T)

    for _ in range(max_scp_iters):
        # Prior residual and Jacobian:
        #   Log(chi_pred^{-1} chi_iter Exp(eta)) ≈ rp + Jp eta.
        rp = inv_retract_SE2(z_pred, z_iter)
        Jp = prior_residual_jacobian_SE2(z_pred, z_iter)

        # GPS residual and Jacobian:
        #   y - h(chi_iter Exp(eta)) ≈ ry - H eta.
        ry = y_gps - gps_measurement_model(z_iter)
        H = gps_measurement_jacobian_intrinsic(z_iter)

        step = solve_trust_region_qp(
            Jp=Jp,
            Pinv=Pinv,
            rp=rp,
            H=H,
            Ninv=Ninv,
            ry=ry,
            trust_radius=trust_radius,
        )
        z_iter = retract_SE2(z_iter, step)

        if np.linalg.norm(step) < tol:
            break

    z_upd = z_iter.copy()

    # Gauss-Newton posterior covariance at the final iterate.
    Jp = prior_residual_jacobian_SE2(z_pred, z_upd)
    H = gps_measurement_jacobian_intrinsic(z_upd)
    Hess = Jp.T @ Pinv @ Jp + H.T @ Ninv @ H
    Hess = 0.5 * (Hess + Hess.T)
    P_upd = np.linalg.inv(Hess + 1e-12 * np.eye(3))
    P_upd = 0.5 * (P_upd + P_upd.T)

    return z_upd, P_upd
