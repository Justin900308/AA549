"""Continuous-discrete ISCVX EKF for flat-earth navigation.

Prediction is identical to the LIEKF.  The discrete landmark conditioning step is
replaced by the intrinsic convexified optimization

    min_chi ||Log(chi chi_pred^{-1})||_{P_pred^{-1}}^2
          + ||y - h(chi)||_{N^{-1}}^2,

solved by successive convexification with CVXPY.  The retraction is the same
right-invariant SE_2(3) retraction used by the paper's IEKF:

    chi = Exp(eta) chi_iter.
"""

from __future__ import annotations

import numpy as np
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CVXPY_AVAILABLE = False

from dynamics import (
    State,
    block_diag_rotation_noise,
    iekf_A_matrix,
    iekf_Q_hat,
    iscvx_measurement_jacobian,
    iscvx_prior_residual_jacobian,
    measurement_model,
    propagate_state,
    right_inv_retract,
    right_retract,
)
from integrator import covariance_euler, spd_project


class ContinuousDiscreteNavISCVX:
    """LIEKF prediction plus intrinsic CVXPY/SCP conditioning."""

    def __init__(
        self,
        state0: State,
        P0,
        Q_base,
        obs_cov,
        dt: float,
        landmarks,
        gravity,
        trust_radius: float = 0.7,
        max_scp_iters: int = 5,
        tol: float = 1e-8,
        solver: str | None = None,
    ) -> None:
        self.state = state0.copy()
        self.P = np.asarray(P0, dtype=float).reshape(9, 9).copy()
        self.Q_base = np.asarray(Q_base, dtype=float).reshape(9, 9)
        self.obs_cov = np.asarray(obs_cov, dtype=float).reshape(3, 3)
        self.dt = float(dt)
        self.landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 3)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.trust_radius = float(trust_radius)
        self.max_scp_iters = int(max_scp_iters)
        self.tol = float(tol)
        self.solver = solver
        self.used_cvxpy = False

    def predict(self, omega) -> None:
        A = iekf_A_matrix(self.gravity)
        Qhat = iekf_Q_hat(self.state, self.Q_base)
        self.state = propagate_state(self.state, omega, self.dt, self.gravity)
        self.P = covariance_euler(self.P, A, Qhat, self.dt)
        self.P = spd_project(self.P, 1e-14)

    def update(self, y) -> None:
        Nhat = block_diag_rotation_noise(self.state, self.obs_cov, len(self.landmarks))
        self.state, self.P, used = intrinsic_cvxpy_update_se23(
            state_pred=self.state,
            P_pred=self.P,
            y=np.asarray(y, dtype=float).reshape(-1),
            N=Nhat,
            landmarks=self.landmarks,
            trust_radius=self.trust_radius,
            max_scp_iters=self.max_scp_iters,
            tol=self.tol,
            solver=self.solver,
        )
        self.used_cvxpy = self.used_cvxpy or used

    def step(self, omega, y=None) -> State:
        self.predict(omega)
        if y is not None:
            self.update(y)
        return self.state.copy()


def _safe_inverse_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = spd_project(np.asarray(A, dtype=float), eps)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return 0.5 * ((vecs * (1.0 / vals)) @ vecs.T + ((vecs * (1.0 / vals)) @ vecs.T).T)


def _solve_qp_direct(Hess: np.ndarray, grad: np.ndarray, trust_radius: float) -> np.ndarray:
    """Fallback exact trust-region solve for the small convex QP."""
    Hess = spd_project(Hess, 1e-12)
    grad = np.asarray(grad, dtype=float).reshape(-1)
    eta_unc = -np.linalg.solve(Hess, grad)
    if np.linalg.norm(eta_unc) <= trust_radius:
        return eta_unc
    I = np.eye(Hess.shape[0])
    lo, hi = 0.0, 1.0
    while np.linalg.norm(-np.linalg.solve(Hess + hi * I, grad)) > trust_radius:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        eta = -np.linalg.solve(Hess + mid * I, grad)
        if np.linalg.norm(eta) > trust_radius:
            lo = mid
        else:
            hi = mid
    return -np.linalg.solve(Hess + hi * I, grad)


def solve_scp_subproblem_cvxpy(rp, Jp, Pinv, ry, H, Ninv, trust_radius, solver=None) -> tuple[np.ndarray, bool]:
    """Solve one convexified ISCVX subproblem.

    Returns (eta, used_cvxpy).  The direct trust-region fallback is only used if
    CVXPY is unavailable or all installed solvers fail.
    """
    rp = np.asarray(rp, dtype=float).reshape(-1)
    ry = np.asarray(ry, dtype=float).reshape(-1)
    Jp = np.asarray(Jp, dtype=float)
    H = np.asarray(H, dtype=float)
    Pinv = spd_project(Pinv, 1e-10)
    Ninv = spd_project(Ninv, 1e-10)

    # Hessian/gradient for fallback and also for checking direct answer.
    Hess = Jp.T @ Pinv @ Jp + H.T @ Ninv @ H
    grad = Jp.T @ Pinv @ rp - H.T @ Ninv @ ry

    if CVXPY_AVAILABLE:
        eta = cp.Variable(Jp.shape[1])
        objective = cp.quad_form(rp + Jp @ eta, cp.psd_wrap(Pinv)) + cp.quad_form(
            ry - H @ eta, cp.psd_wrap(Ninv)
        )
        constraints = [cp.sum_squares(eta) <= float(trust_radius) ** 2]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        installed = set(cp.installed_solvers())
        candidates = []
        if solver is not None:
            candidates.append(solver)
        candidates += ["CLARABEL", "SCS", "OSQP", "SCIPY"]
        for name in candidates:
            if name not in installed:
                continue
            try:
                kwargs = {}
                if name == "SCS":
                    kwargs.update({"eps": 1e-5, "max_iters": 5000})
                if name == "OSQP":
                    # OSQP cannot handle the quadratic norm constraint, but may work
                    # if CVXPY internally transforms the problem in future versions.
                    kwargs.update({"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 10000})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob.solve(solver=name, verbose=False, **kwargs)
                if eta.value is not None and prob.status in {"optimal", "optimal_inaccurate"}:
                    return np.asarray(eta.value, dtype=float).reshape(-1), True
            except Exception:
                pass

    return _solve_qp_direct(Hess, grad, trust_radius), False


def intrinsic_cvxpy_update_se23(
    state_pred: State,
    P_pred: np.ndarray,
    y: np.ndarray,
    N: np.ndarray,
    landmarks: np.ndarray,
    trust_radius: float = 0.7,
    max_scp_iters: int = 5,
    tol: float = 1e-8,
    solver: str | None = None,
) -> tuple[State, np.ndarray, bool]:
    state_iter = state_pred.copy()
    Pinv = _safe_inverse_spd(P_pred, 1e-12)
    Ninv = _safe_inverse_spd(N, 1e-12)
    used_cvxpy_any = False

    for _ in range(max_scp_iters):
        rp = right_inv_retract(state_pred, state_iter)
        Jp = iscvx_prior_residual_jacobian(state_pred, state_iter)
        ry = y - measurement_model(state_iter, landmarks)
        H = iscvx_measurement_jacobian(state_iter, landmarks)

        eta, used_cvxpy = solve_scp_subproblem_cvxpy(
            rp, Jp, Pinv, ry, H, Ninv, trust_radius, solver=solver
        )
        used_cvxpy_any = used_cvxpy_any or used_cvxpy
        state_iter = right_retract(state_iter, eta)
        if np.linalg.norm(eta) < tol:
            break

    state_upd = state_iter.copy()
    Jp = iscvx_prior_residual_jacobian(state_pred, state_upd)
    H = iscvx_measurement_jacobian(state_upd, landmarks)
    Hess = Jp.T @ Pinv @ Jp + H.T @ Ninv @ H
    P_upd = np.linalg.inv(spd_project(Hess, 1e-10))
    P_upd = spd_project(P_upd, 1e-14)
    return state_upd, P_upd, used_cvxpy_any
