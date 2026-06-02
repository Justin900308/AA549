"""CVXPY-backed continuous-discrete intrinsic/SCP filter for the simplified car.

This version replaces the hard trust-region constraint in the conditioning
subproblem with an L2 step regularizer and a backtracking-style acceptance test.
The prediction is exactly the same as the LIEKF prediction,

    z_pred = flow(z_hat, u),
    Pdot   = A_liekf P + P A_liekf^T + Q,

and only the discrete GPS conditioning step is replaced by a CVXPY/SCP solve.
At each SCP iteration, with chi = chi_iter Exp(eta), the convex subproblem is

    min_eta ||rp + Jp eta||_{P_pred^{-1}}^2
          + ||ry - H eta||_{N^{-1}}^2
          + 0.5 * rho * ||eta||_2^2
    s.t.    obs_r^2 - ||p_iter - obs_j||_2^2
              - 2 (p_iter - obs_j)^T (H_p eta) <= 0,   for all obstacles j.

The hard constraint

    ||eta||_2 <= trust_radius

is removed.  Instead, rho is increased by ``l2_reg_growth`` whenever the trial
state increases the true nonlinear conditioning objective

    ||Log(z_pred^{-1} z)||_{P_pred^{-1}}^2
    + ||y_gps - h(z)||_{N^{-1}}^2.

This is a soft trust-region / Levenberg-Marquardt-like version of the original
SCP conditioning step: fewer constraints, but the step is still damped when the
local linearization is poor.
"""

from __future__ import annotations

import numpy as np
import time
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
    unicycle_dynamics,
    retract_SE2,
    wrap_angle,
)
from integrator import covariance_euler, rk4


class ContinuousDiscreteCarISCVXCVXPY:
    """LIEKF prediction + CVXPY intrinsic convexified GPS conditioning on SE(2).

    This version uses a soft L2 step regularizer instead of a hard trust-region
    constraint.  Obstacle constraints are still kept as linearized half-spaces.
    If ``obs`` is empty, the subproblem has no constraints at all.
    """

    def __init__(
        self,
        z0,
        P0,
        Q,
        N,
        dt: float,
        flag: bool,
        l2_reg_initial: float = 1e-4,
        l2_reg_growth: float = 10.0,
        max_backtracking: int = 8,
        max_scp_iters: int = 5,
        tol: float = 1e-9,
        solver: str | None = None,
        fallback_without_cvxpy: bool = False,
        obs: np.ndarray | None = None,
        obs_r: float = 0.0,
    ) -> None:
        self.z = np.asarray(z0, dtype=float).reshape(3).copy()
        self.z[0] = wrap_angle(self.z[0])
        self.P = np.asarray(P0, dtype=float).reshape(3, 3).copy()
        self.Q = np.asarray(Q, dtype=float).reshape(3, 3)
        self.N = np.asarray(N, dtype=float).reshape(2, 2)
        self.dt = float(dt)
        self.l2_reg_initial = float(l2_reg_initial)
        self.l2_reg_growth = float(l2_reg_growth)
        self.max_backtracking = int(max_backtracking)
        self.max_scp_iters = int(max_scp_iters)
        self.tol = float(tol)
        self.solver = solver
        # Kept only for backward-compatible construction.  This version always
        # uses CVXPY because the conditioning step may include obstacle constraints.
        self.fallback_without_cvxpy = bool(fallback_without_cvxpy)
        self.flag = flag
        self.obs = _format_obstacles(obs)
        self.obs_r = float(obs_r)
        self.used_cvxpy = False
        self.used_fallback = False

        # Diagnostics from the most recent update.
        self.last_l2_reg_weight = self.l2_reg_initial
        self.last_backtracking_steps = 0
        self.last_rejected_step = False
        self.update_t = None
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required because the ISCVX conditioning step is solved "
                "as a CVXPY problem. Install it with `pip install cvxpy`."
            )
        if self.l2_reg_initial < 0.0:
            raise ValueError("l2_reg_initial must be nonnegative.")
        if self.l2_reg_growth <= 1.0:
            raise ValueError("l2_reg_growth must be greater than 1.0.")
        if self.max_backtracking < 1:
            raise ValueError("max_backtracking must be at least 1.")

    def predict(self, u) -> None:
        """Same continuous-discrete prediction step as the LIEKF."""
        u = np.asarray(u, dtype=float).reshape(2)
        v, omega = float(u[0]), float(u[1])
        A = liekf_A_matrix(v, omega)

        self.z = rk4(unicycle_dynamics, self.z, u, self.dt)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def update(self, y_gps) -> None:
        t0 = time.time()
        self.z, self.P, used_cvxpy, info = intrinsic_cvxpy_update_SE2(
            z_pred=self.z,
            P_pred=self.P,
            y_gps=np.asarray(y_gps, dtype=float).reshape(2),
            N=self.N,
            l2_reg_initial=self.l2_reg_initial,
            l2_reg_growth=self.l2_reg_growth,
            max_backtracking=self.max_backtracking,
            max_scp_iters=self.max_scp_iters,
            tol=self.tol,
            solver=self.solver,
            fallback_without_cvxpy=self.fallback_without_cvxpy,
            flag = self.flag,
            obs=self.obs,
            obs_r=self.obs_r,
        )
        self.used_cvxpy = self.used_cvxpy or used_cvxpy
        self.used_fallback = self.used_fallback or (not used_cvxpy)
        self.last_l2_reg_weight = info["final_l2_reg_weight"]
        self.last_backtracking_steps = info["total_backtracking_steps"]
        self.last_rejected_step = info["rejected_step"]
        tf = time.time()
        self.update_t = tf-t0

    def step(self, u, y_gps=None) -> np.ndarray:
        self.predict(u)
        if y_gps is not None:
            self.update(y_gps)
        return self.z.copy()


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------


def _format_obstacles(obs: np.ndarray | None) -> np.ndarray:
    """Return obstacles as an array with shape (num_obs, 2)."""
    if obs is None:
        return np.zeros((0, 2), dtype=float)
    obs_arr = np.asarray(obs, dtype=float)
    if obs_arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    return obs_arr.reshape(-1, 2)


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


def _obstacle_constraints(
    eta,
    z_lin: np.ndarray,
    flag: bool,
    obs: np.ndarray | None,
    obs_r: float,
    position_jacobian: np.ndarray | None,
) -> list:
    """Build linearized obstacle constraints for one SCP subproblem.

    For each obstacle center obs_j, the nonlinear safe-set constraint is

        obs_r^2 - ||p - obs_j||^2 <= 0.

    Linearizing at p_iter = z_lin[1:3] gives

        h_j + a_j^T delta_p <= 0,

    where

        h_j = obs_r^2 - ||p_iter - obs_j||^2,
        a_j = -2 (p_iter - obs_j),
        delta_p ≈ H_p eta.

    If ``position_jacobian`` is None, the code uses the direct additive form
    delta_p = eta[1:3].
    """
    obs_arr = _format_obstacles(obs)

    z_lin = np.asarray(z_lin, dtype=float).reshape(3)
    p_iter = z_lin[1:3]
    r = float(obs_r)

    if position_jacobian is None:
        delta_p = eta[1:3]
    else:
        H_p = np.asarray(position_jacobian, dtype=float).reshape(2, -1)
        delta_p = H_p @ eta
    constraints = []
    if flag == True:
        for obs_j in obs_arr:
            diff = p_iter - obs_j
            h_j = r**2 - diff @ diff
            a_j = -2.0 * diff
            lhs = h_j + a_j @ delta_p
            constraints.append(lhs <= 0.0)

    return constraints


# -----------------------------------------------------------------------------
# CVXPY conditioning solve
# -----------------------------------------------------------------------------


def solve_regularized_qp_cvxpy(
    Jp: np.ndarray,
    Pinv: np.ndarray,
    rp: np.ndarray,
    H: np.ndarray,
    Ninv: np.ndarray,
    ry: np.ndarray,
    l2_reg_weight: float,
    solver: str | None = None,
    z_lin: np.ndarray | None = None,
    flag: bool = True,
    obs: np.ndarray | None = None,
    obs_r: float = 0.0,
    use_position_jacobian_for_obstacles: bool = True,
) -> np.ndarray:
    """Solve the L2-regularized SCP QP with CVXPY.

    The hard trust-region constraint is removed.  The subproblem is

        min_eta ||rp + Jp eta||_Pinv^2
              + ||ry - H eta||_Ninv^2
              + 0.5 * l2_reg_weight * ||eta||_2^2

    subject only to the linearized obstacle constraints, if any.
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("cvxpy is not installed. Install it with `pip install cvxpy`.")

    Jp = np.asarray(Jp, dtype=float)
    H = np.asarray(H, dtype=float)
    rp = np.asarray(rp, dtype=float).reshape(-1)
    ry = np.asarray(ry, dtype=float).reshape(-1)
    Pinv = _make_psd(Pinv)
    Ninv = _make_psd(Ninv)
    obs_arr = _format_obstacles(obs)

    if z_lin is None:
        raise ValueError("z_lin must be provided for the conditioning problem.")

    rho = float(l2_reg_weight)
    if rho < 0.0:
        raise ValueError("l2_reg_weight must be nonnegative.")

    dim = Jp.shape[1]
    eta = cp.Variable(dim)
    objective_expr = (
        cp.quad_form(rp + Jp @ eta, cp.psd_wrap(Pinv))
        + cp.quad_form(ry - H @ eta, cp.psd_wrap(Ninv))
        + 0.5 * rho * cp.sum_squares(eta)
    )

    # Only obstacle half-spaces remain as hard constraints.  If obs is empty,
    # this list is empty and the subproblem is an unconstrained regularized QP.
    H_p = H if use_position_jacobian_for_obstacles else None
    constraints = _obstacle_constraints(
        eta=eta,
        z_lin=np.asarray(z_lin, dtype=float).reshape(3),
        flag = flag,
        obs=obs_arr,
        obs_r=float(obs_r),
        position_jacobian=H_p,
    )

    problem = cp.Problem(cp.Minimize(objective_expr), constraints)

    installed = set(cp.installed_solvers())
    errors: list[str] = []

    # Without the norm trust-region, this is a QP with affine constraints, so
    # OSQP/HIGHS/SCIPY are good first choices when installed.
    candidates = []
    if solver is not None:
        candidates.append(solver)
    candidates += ["OSQP", "HIGHS", "SCIPY", "CLARABEL", "MOSEK", "SCS"]

    for solver_name in candidates:
        if solver_name not in installed:
            continue
        try:
            kwargs = {}
            if solver_name == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 5000})
            problem.solve(solver=solver_name, verbose=False, **kwargs)
            if eta.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
                return np.asarray(eta.value, dtype=float).reshape(dim)
        except Exception as exc:
            errors.append(f"regularized {solver_name}: {exc}")

    msg = "CVXPY failed to solve the intrinsic L2-regularized update."
    if errors:
        msg += " Solver errors: " + " | ".join(errors)
    raise RuntimeError(msg)


def conditioning_objective_SE2(
    z_pred: np.ndarray,
    z: np.ndarray,
    y_gps: np.ndarray,
    Pinv: np.ndarray,
    Ninv: np.ndarray,
) -> float:
    """Evaluate the true nonlinear conditioning objective at a state estimate.

    This is the objective used both for the SCP stopping condition and for the
    backtracking acceptance test:

        ||Log(z_pred^{-1} z)||_{P_pred^{-1}}^2
        + ||y_gps - h(z)||_{N^{-1}}^2.
    """
    rp = inv_retract_SE2(
        np.asarray(z_pred, dtype=float).reshape(3),
        np.asarray(z, dtype=float).reshape(3),
    )
    ry = np.asarray(y_gps, dtype=float).reshape(2) - gps_measurement_model(z)
    return float(rp.T @ Pinv @ rp + ry.T @ Ninv @ ry)


# -----------------------------------------------------------------------------
# Full intrinsic SCP update
# -----------------------------------------------------------------------------


def intrinsic_cvxpy_update_SE2(
    z_pred: np.ndarray,
    P_pred: np.ndarray,
    y_gps: np.ndarray,
    N: np.ndarray,
    l2_reg_initial: float = 1e-4,
    l2_reg_growth: float = 10.0,
    max_backtracking: int = 8,
    max_scp_iters: int = 10,
    tol: float = 1e-9,
    solver: str | None = None,
    fallback_without_cvxpy: bool = False,
    flag: bool = True,
    obs: np.ndarray | None = None,
    obs_r: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, bool, dict]:
    """Successive convexified intrinsic GPS update on SE(2), CVXPY version.

    Returns
    -------
    z_upd, P_upd, used_cvxpy, info
        ``used_cvxpy`` is always True for this CVXPY version.  ``info`` stores
        the final L2 regularization weight and backtracking diagnostics.
    """
    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    z_iter = z_pred.copy()
    P_pred = np.asarray(P_pred, dtype=float).reshape(3, 3)
    N = np.asarray(N, dtype=float).reshape(2, 2)
    obs_arr = _format_obstacles(obs)
    obs_r = float(obs_r)

    Pinv = _safe_inverse_spd(P_pred, eps=1e-10)
    Ninv = _safe_inverse_spd(N, eps=1e-10)

    if not CVXPY_AVAILABLE:
        raise ImportError(
            "cvxpy is required because the ISCVX conditioning step is solved "
            "as a CVXPY problem. Install it with `pip install cvxpy`."
        )
    if l2_reg_initial < 0.0:
        raise ValueError("l2_reg_initial must be nonnegative.")
    if l2_reg_growth <= 1.0:
        raise ValueError("l2_reg_growth must be greater than 1.0.")
    if max_backtracking < 1:
        raise ValueError("max_backtracking must be at least 1.")

    used_cvxpy = True
    rho = float(l2_reg_initial)
    total_backtracking_steps = 0
    rejected_step = False

    prev_obj = conditioning_objective_SE2(
        z_pred=z_pred,
        z=z_iter,
        y_gps=y_gps,
        Pinv=Pinv,
        Ninv=Ninv,
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

        accepted = False
        trial_rho = max(rho, 0.0)
        trial_step = None
        trial_z = None
        trial_obj = np.inf

        for bt in range(max_backtracking):
            step = solve_regularized_qp_cvxpy(
                Jp=Jp,
                Pinv=Pinv,
                rp=rp,
                H=H,
                Ninv=Ninv,
                ry=ry,
                l2_reg_weight=trial_rho,
                solver=solver,
                z_lin=z_iter,
                flag = flag,
                obs=obs_arr,
                obs_r=obs_r,
                use_position_jacobian_for_obstacles=True,
            )

            z_candidate = retract_SE2(z_iter, step)
            curr_obj = conditioning_objective_SE2(
                z_pred=z_pred,
                z=z_candidate,
                y_gps=y_gps,
                Pinv=Pinv,
                Ninv=Ninv,
            )

            # Accept a monotone step.  The small tolerance avoids rejecting a
            # numerically identical objective value.
            accept_tol = float(tol) * (1.0 + abs(prev_obj))
            if curr_obj <= prev_obj + accept_tol:
                accepted = True
                trial_step = step
                trial_z = z_candidate
                trial_obj = curr_obj
                total_backtracking_steps += bt
                break

            # Nonlinear objective increased, so damp harder and re-solve.
            trial_rho *= float(l2_reg_growth)

        if not accepted:
            # A monotone step was not found.  Do not accept an objective-increasing
            # update; stop the SCP loop and keep the current iterate.
            rejected_step = True
            break

        z_iter = trial_z
        rho = trial_rho

        obj_diff = abs(trial_obj - prev_obj)
        obj_scale = 1.0 + abs(prev_obj)
        if obj_diff <= float(tol) * obj_scale or np.linalg.norm(trial_step) <= float(tol):
            prev_obj = trial_obj
            break
        prev_obj = trial_obj

    z_upd = z_iter.copy()

    # Gauss-Newton posterior covariance at the final iterate.  The L2
    # regularizer is a numerical globalization device, so it is not included in
    # the covariance Hessian below.  Include +rho*I here only if you want the
    # covariance to reflect the artificial damping as an extra prior.
    Jp = prior_residual_jacobian_SE2(z_pred, z_upd)
    H = gps_measurement_jacobian_intrinsic(z_upd)
    Hess = Jp.T @ Pinv @ Jp + H.T @ Ninv @ H
    Hess = 0.5 * (Hess + Hess.T)
    P_upd = np.linalg.inv(Hess + 1e-12 * np.eye(3))
    P_upd = 0.5 * (P_upd + P_upd.T)

    info = {
        "final_l2_reg_weight": rho,
        "total_backtracking_steps": total_backtracking_steps,
        "rejected_step": rejected_step,
        "final_objective": prev_obj,
    }

    return z_upd, P_upd, used_cvxpy, info
