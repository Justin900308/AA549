"""CVXPY-backed continuous-discrete intrinsic/SCP filter for the simplified car.

This is the more general version of ``continuous_discrete_iscvx.py``.  The
prediction is exactly the same as the LIEKF prediction,

    z_pred = flow(z_hat, u),
    Pdot   = A_liekf P + P A_liekf^T + Q,

and only the discrete GPS conditioning step is replaced by a CVXPY/SCP solve.
At each SCP iteration, with chi = chi_iter Exp(eta), the convex subproblem is

    min_eta ||rp + Jp eta||_{P_pred^{-1}}^2
          + ||ry - H eta||_{N^{-1}}^2
    s.t.    ||eta||_2 <= trust_radius.

The code uses CVXPY when it is installed.  If CVXPY is not installed, the class
can optionally fall back to the closed-form trust-region solver from
``continuous_discrete_iscvx.py`` so the example still runs.  Set
``fallback_without_cvxpy=False`` to require CVXPY strictly.
"""

from __future__ import annotations

import warnings
import numpy as np

try:  # optional dependency
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when cvxpy is absent
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

# Fallback keeps this file runnable on machines that do not have cvxpy.
from continuous_discrete_iscvx import solve_trust_region_qp as solve_trust_region_qp_direct


class ContinuousDiscreteCarISCVXCVXPY:
    """LIEKF prediction + CVXPY intrinsic convexified GPS conditioning on SE(2)."""

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
        fallback_without_cvxpy: bool = True,
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
        self.fallback_without_cvxpy = bool(fallback_without_cvxpy)
        self.used_cvxpy = False
        self.used_fallback = False

        if not CVXPY_AVAILABLE and not self.fallback_without_cvxpy:
            raise ImportError(
                "cvxpy is required for ContinuousDiscreteCarISCVXCVXPY when "
                "fallback_without_cvxpy=False. Install it with `pip install cvxpy`."
            )

    def predict(self, u) -> None:
        """Same continuous-discrete prediction step as the LIEKF."""
        u = np.asarray(u, dtype=float).reshape(2)
        v, omega = float(u[0]), float(u[1])
        A = liekf_A_matrix(v, omega)

        self.z = propagate_pose_rk4(self.z, u, self.dt)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def update(self, y_gps) -> None:
        self.z, self.P, used_cvxpy = intrinsic_cvxpy_update_SE2(
            z_pred=self.z,
            P_pred=self.P,
            y_gps=np.asarray(y_gps, dtype=float).reshape(2),
            N=self.N,
            trust_radius=self.trust_radius,
            max_scp_iters=self.max_scp_iters,
            tol=self.tol,
            solver=self.solver,
            fallback_without_cvxpy=self.fallback_without_cvxpy,
        )
        self.used_cvxpy = self.used_cvxpy or used_cvxpy
        self.used_fallback = self.used_fallback or (not used_cvxpy)

    def step(self, u, y_gps=None) -> np.ndarray:
        self.predict(u)
        if y_gps is not None:
            self.update(y_gps)
        return self.z.copy()


def _project_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to the SPD cone by eigenvalue clipping."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    A_spd = (vecs * vals) @ vecs.T
    return 0.5 * (A_spd + A_spd.T)


def _safe_inverse_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return inv(project_spd(A)) without amplifying tiny/negative eigenvalues."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    Ainv = (vecs * (1.0 / vals)) @ vecs.T
    return 0.5 * (Ainv + Ainv.T)


def _make_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize and project a weight matrix to the SPD cone for quad_form."""
    return _project_spd(A, eps=eps)


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
    """Solve the SCP trust-region QP with CVXPY.

    The function first solves the unconstrained convex QP with a QP solver.  If
    the unconstrained minimizer is inside the trust region, it is also the
    trust-region solution.  If not, the function solves the conic/QCQP form with
    the trust-region constraint.  This two-stage strategy is more numerically
    reliable for the paper's nearly-known initial position covariance, while
    still supporting a true trust region for harder cases.
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("cvxpy is not installed. Install it with `pip install cvxpy`.")

    Jp = np.asarray(Jp, dtype=float)
    H = np.asarray(H, dtype=float)
    rp = np.asarray(rp, dtype=float).reshape(-1)
    ry = np.asarray(ry, dtype=float).reshape(-1)
    Pinv = _make_psd(Pinv)
    Ninv = _make_psd(Ninv)

    dim = Jp.shape[1]
    eta = cp.Variable(dim)
    objective_expr = (
        cp.quad_form(rp + Jp @ eta, cp.psd_wrap(Pinv))
        + cp.quad_form(ry - H @ eta, cp.psd_wrap(Ninv))
    )

    installed = set(cp.installed_solvers())
    errors: list[str] = []

    # Stage 1: unconstrained QP.  This is often sufficient and lets OSQP/HIGHS
    # handle the problem without SOC/QCQP numerical issues.
    unconstrained = cp.Problem(cp.Minimize(objective_expr))
    qp_candidates = []
    if solver is not None:
        qp_candidates.append(solver)
    qp_candidates += ["OSQP", "HIGHS", "CLARABEL", "SCIPY", "SCS"]

    for solver_name in qp_candidates:
        if solver_name not in installed:
            continue
        try:
            kwargs = {}
            if solver_name == "OSQP":
                kwargs.update({"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 10000})
            if solver_name == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 5000})
            unconstrained.solve(solver=solver_name, verbose=False, **kwargs)
            if eta.value is not None and unconstrained.status in {"optimal", "optimal_inaccurate"}:
                step = np.asarray(eta.value, dtype=float).reshape(dim)
                if np.linalg.norm(step) <= float(trust_radius) * (1.0 + 1e-8):
                    return step
                break
        except Exception as exc:
            errors.append(f"unconstrained {solver_name}: {exc}")

    # Stage 2: trust-region constrained problem.  This is a convex QCQP/SOCP.
    eta = cp.Variable(dim)
    constrained_objective = (
        cp.quad_form(rp + Jp @ eta, cp.psd_wrap(Pinv))
        + cp.quad_form(ry - H @ eta, cp.psd_wrap(Ninv))
    )
    constraints = [cp.sum_squares(eta) <= float(trust_radius) ** 2]
    constrained = cp.Problem(cp.Minimize(constrained_objective), constraints)

    conic_candidates = []
    if solver is not None:
        conic_candidates.append(solver)
    conic_candidates += ["CLARABEL", "MOSEK", "SCS", "SCIPY"]

    for solver_name in conic_candidates:
        if solver_name not in installed:
            continue
        try:
            kwargs = {}
            if solver_name == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 5000})
            constrained.solve(solver=solver_name, verbose=False, **kwargs)
            if eta.value is not None and constrained.status in {"optimal", "optimal_inaccurate"}:
                return np.asarray(eta.value, dtype=float).reshape(dim)
        except Exception as exc:
            errors.append(f"constrained {solver_name}: {exc}")

    msg = "CVXPY failed to solve the intrinsic trust-region update."
    if errors:
        msg += " Solver errors: " + " | ".join(errors)
    raise RuntimeError(msg)


def intrinsic_cvxpy_update_SE2(
    z_pred: np.ndarray,
    P_pred: np.ndarray,
    y_gps: np.ndarray,
    N: np.ndarray,
    trust_radius: float = 0.5,
    max_scp_iters: int = 5,
    tol: float = 1e-9,
    solver: str | None = None,
    fallback_without_cvxpy: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Successive convexified intrinsic GPS update on SE(2), CVXPY version.

    Returns
    -------
    z_upd, P_upd, used_cvxpy
        ``used_cvxpy`` is False only when cvxpy is absent and the optional
        direct trust-region fallback was used.
    """
    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    z_iter = z_pred.copy()
    P_pred = np.asarray(P_pred, dtype=float).reshape(3, 3)
    N = np.asarray(N, dtype=float).reshape(2, 2)

    Pinv = _safe_inverse_spd(P_pred, eps=1e-10)
    Ninv = _safe_inverse_spd(N, eps=1e-10)

    used_cvxpy = CVXPY_AVAILABLE
    if not CVXPY_AVAILABLE:
        if not fallback_without_cvxpy:
            raise ImportError("cvxpy is not installed. Install it with `pip install cvxpy`.")
        warnings.warn(
            "cvxpy is not installed; falling back to the direct trust-region solver. "
            "Install cvxpy to use the CVXPY backend.",
            RuntimeWarning,
            stacklevel=2,
        )

    for _ in range(max_scp_iters):
        # Prior residual and Jacobian:
        #   Log(chi_pred^{-1} chi_iter Exp(eta)) ≈ rp + Jp eta.
        rp = inv_retract_SE2(z_pred, z_iter)
        Jp = prior_residual_jacobian_SE2(z_pred, z_iter)

        # GPS residual and Jacobian:
        #   y - h(chi_iter Exp(eta)) ≈ ry - H eta.
        ry = y_gps - gps_measurement_model(z_iter)
        H = gps_measurement_jacobian_intrinsic(z_iter)

        if CVXPY_AVAILABLE:
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
        else:
            step = solve_trust_region_qp_direct(
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

    return z_upd, P_upd, used_cvxpy
