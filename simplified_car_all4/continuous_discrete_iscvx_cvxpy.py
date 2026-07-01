"""Continuous-discrete ISCVX filter for the simplified car.

Prediction is identical to the LIEKF.  The discrete GPS conditioning step is
replaced by an intrinsic successive-convexification/CVXPY update on SE(2).
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CVXPY_AVAILABLE = False

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


def _project_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    A_spd = (vecs * vals) @ vecs.T
    return 0.5 * (A_spd + A_spd.T)


def _safe_inverse_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    Ainv = (vecs * (1.0 / vals)) @ vecs.T
    return 0.5 * (Ainv + Ainv.T)


def solve_trust_region_qp_cvxpy(
    Jp: np.ndarray,
    Pinv: np.ndarray,
    rp: np.ndarray,
    H: np.ndarray,
    Ninv: np.ndarray,
    ry: np.ndarray,
    trust_radius: float,
    solver: str | None = None,
) -> np.ndarray:
    """Solve the small convex SCP trust-region QP with CVXPY.

    The unconstrained convex QP is solved first.  If that minimizer is inside
    the trust region, it is the trust-region solution.  Otherwise the full
    conic/QCQP problem with ``||eta||_2 <= trust_radius`` is solved.
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("cvxpy is not installed. Install with `pip install cvxpy`.")

    Jp = np.asarray(Jp, dtype=float)
    H = np.asarray(H, dtype=float)
    rp = np.asarray(rp, dtype=float).reshape(-1)
    ry = np.asarray(ry, dtype=float).reshape(-1)
    Pinv = _project_spd(Pinv)
    Ninv = _project_spd(Ninv)
    dim = Jp.shape[1]
    installed = set(cp.installed_solvers())
    errors = []

    def make_objective(var):
        return (
            cp.quad_form(rp + Jp @ var, cp.psd_wrap(Pinv))
            + cp.quad_form(ry - H @ var, cp.psd_wrap(Ninv))
        )

    # Stage 1: unconstrained QP.
    eta = cp.Variable(dim)
    unconstrained = cp.Problem(cp.Minimize(make_objective(eta)))
    candidates_qp = []
    if solver is not None:
        candidates_qp.append(solver)
    candidates_qp += ["OSQP", "HIGHS", "CLARABEL", "SCIPY", "SCS"]

    for solver_name in candidates_qp:
        if solver_name not in installed:
            continue
        try:
            kwargs = {}
            if solver_name == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 5000})
            if solver_name == "OSQP":
                kwargs.update({"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 10000})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                unconstrained.solve(solver=solver_name, verbose=False, **kwargs)
            if eta.value is not None and unconstrained.status in {"optimal", "optimal_inaccurate"}:
                step = np.asarray(eta.value, dtype=float).reshape(dim)
                if np.linalg.norm(step) <= float(trust_radius) * (1.0 + 1e-8):
                    return step
                break
        except Exception as exc:
            errors.append(f"unconstrained {solver_name}: {exc}")

    # Stage 2: constrained conic/QCQP.
    eta = cp.Variable(dim)
    constrained = cp.Problem(
        cp.Minimize(make_objective(eta)),
        [cp.sum_squares(eta) <= float(trust_radius) ** 2],
    )
    candidates_conic = []
    if solver is not None:
        candidates_conic.append(solver)
    candidates_conic += ["CLARABEL", "SCS", "SCIPY"]

    for solver_name in candidates_conic:
        if solver_name not in installed:
            continue
        try:
            kwargs = {}
            if solver_name == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 5000})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                constrained.solve(solver=solver_name, verbose=False, **kwargs)
            if eta.value is not None and constrained.status in {"optimal", "optimal_inaccurate"}:
                return np.asarray(eta.value, dtype=float).reshape(dim)
        except Exception as exc:
            errors.append(f"constrained {solver_name}: {exc}")

    raise RuntimeError("CVXPY failed to solve ISCVX subproblem: " + " | ".join(errors))


class ContinuousDiscreteCarISCVXCVXPY:
    """LIEKF prediction + CVXPY intrinsic GPS conditioning."""

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
        solver: str | None = None,
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
        self.solver = solver
        self.used_cvxpy = False

    def predict(self, u) -> None:
        u = np.asarray(u, dtype=float).reshape(2)
        v, omega = float(u[0]), float(u[1])
        A = liekf_A_matrix(v, omega)
        self.z = propagate_pose_rk4(self.z, u, self.dt)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def update(self, y_gps) -> None:
        self.z, self.P = intrinsic_cvxpy_update_SE2(
            z_pred=self.z,
            P_pred=self.P,
            y_gps=np.asarray(y_gps, dtype=float).reshape(2),
            N=self.N,
            trust_radius=self.trust_radius,
            max_scp_iters=self.max_scp_iters,
            tol=self.tol,
            solver=self.solver,
        )
        self.used_cvxpy = True

    def step(self, u, y_gps=None) -> np.ndarray:
        self.predict(u)
        if y_gps is not None:
            self.update(y_gps)
        return self.z.copy()


def intrinsic_cvxpy_update_SE2(
    z_pred: np.ndarray,
    P_pred: np.ndarray,
    y_gps: np.ndarray,
    N: np.ndarray,
    trust_radius: float = 0.5,
    max_scp_iters: int = 5,
    tol: float = 1e-9,
    solver: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Successive convexified intrinsic GPS update on SE(2)."""
    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    z_iter = z_pred.copy()
    P_pred = np.asarray(P_pred, dtype=float).reshape(3, 3)
    N = np.asarray(N, dtype=float).reshape(2, 2)

    Pinv = _safe_inverse_spd(P_pred, eps=1e-10)
    Ninv = _safe_inverse_spd(N, eps=1e-10)

    for _ in range(max_scp_iters):
        # Prior residual:
        # Log(chi_pred^{-1} chi_iter Exp(eta)) ≈ rp + Jp eta.
        rp = inv_retract_SE2(z_pred, z_iter)
        Jp = prior_residual_jacobian_SE2(z_pred, z_iter)

        # GPS residual:
        # y - h(chi_iter Exp(eta)) ≈ ry - H eta.
        ry = y_gps - gps_measurement_model(z_iter)
        H = gps_measurement_jacobian_intrinsic(z_iter)

        step = solve_trust_region_qp_cvxpy(
            Jp=Jp,
            Pinv=Pinv,
            rp=rp,
            H=H,
            Ninv=Ninv,
            ry=ry,
            trust_radius=trust_radius,
            solver=solver,
        )
        z_iter = retract_SE2(z_iter, step)

        if np.linalg.norm(step) < tol:
            break

    z_upd = z_iter.copy()

    # Gauss-Newton covariance at the final point.
    Jp = prior_residual_jacobian_SE2(z_pred, z_upd)
    H = gps_measurement_jacobian_intrinsic(z_upd)
    Hess = Jp.T @ Pinv @ Jp + H.T @ Ninv @ H
    Hess = 0.5 * (Hess + Hess.T)
    P_upd = np.linalg.inv(Hess + 1e-12 * np.eye(3))
    P_upd = 0.5 * (P_upd + P_upd.T)
    return z_upd, P_upd
