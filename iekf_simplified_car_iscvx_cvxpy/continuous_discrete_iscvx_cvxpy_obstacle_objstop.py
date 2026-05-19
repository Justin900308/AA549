"""CVXPY-backed continuous-discrete intrinsic/SCP filter for the simplified car.

This version always forms the CVXPY conditioning step with the trust-region
constraint and the linearized obstacle-avoidance constraints.  The prediction is
exactly the same as the LIEKF prediction,

    z_pred = flow(z_hat, u),
    Pdot   = A_liekf P + P A_liekf^T + Q,

and only the discrete GPS conditioning step is replaced by a CVXPY/SCP solve.
At each SCP iteration, with chi = chi_iter Exp(eta), the convex subproblem is

    min_eta ||rp + Jp eta||_{P_pred^{-1}}^2
          + ||ry - H eta||_{N^{-1}}^2
    s.t.    ||eta||_2 <= trust_radius,
            obs_r^2 - ||p_iter - obs_j||_2^2
              - 2 (p_iter - obs_j)^T (H_p eta) <= 0,   for all obstacles j.

Here p_iter = z_iter[1:3], and H_p maps the intrinsic perturbation eta to the
first-order position perturbation.  For additive [theta, x, y] coordinates,
H_p eta is simply eta[1:3], matching the form

    h_j = obs_r**2 - ||x_t[1:3] - obs_j||^2
    a   = -2 * (x_t[1:3] - obs_j)
    h_j + a @ d_t[1:3] <= 0.

The conditioning step always uses the constrained CVXPY problem.  Therefore this
version requires CVXPY and does not use the closed-form fallback solver.
"""

from __future__ import annotations

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
    unicycle_dynamics,
    retract_SE2,
    wrap_angle,
)
from integrator import covariance_euler, rk4


class ContinuousDiscreteCarISCVXCVXPY:
    """LIEKF prediction + CVXPY intrinsic convexified GPS conditioning on SE(2).

    Obstacle constraints are passed through ``obs`` and ``obs_r``.  The obstacle
    set is interpreted as circular keep-out regions in the x-y plane with
    centers ``obs[j]`` and radius ``obs_r``.  If ``obs`` is empty, the same
    constrained CVXPY path is still used, but no obstacle half-spaces are added.
    """

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
        self.trust_radius = float(trust_radius)
        self.max_scp_iters = int(max_scp_iters)
        self.tol = float(tol)
        self.solver = solver
        # Kept only for backward-compatible construction.  This version always
        # uses CVXPY because the conditioning step always includes constraints.
        self.fallback_without_cvxpy = bool(fallback_without_cvxpy)
        self.obs = _format_obstacles(obs)
        self.obs_r = float(obs_r)
        self.used_cvxpy = False
        self.used_fallback = False

        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required because the ISCVX conditioning step is solved "
                "as a constrained CVXPY problem. Install it with `pip install cvxpy`."
            )

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
            obs=self.obs,
            obs_r=self.obs_r,
        )
        self.used_cvxpy = self.used_cvxpy or used_cvxpy
        self.used_fallback = self.used_fallback or (not used_cvxpy)

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
        # This is exactly the user's snippet if eta is the additive state step:
        # LHS = h_j + a @ eta[1:3] <= 0.
        delta_p = eta[1:3]
    else:
        H_p = np.asarray(position_jacobian, dtype=float).reshape(2, -1)
        delta_p = H_p @ eta

    constraints = []
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


def solve_trust_region_qp_cvxpy(
    Jp: np.ndarray,
    Pinv: np.ndarray,
    rp: np.ndarray,
    H: np.ndarray,
    Ninv: np.ndarray,
    ry: np.ndarray,
    trust_radius: float,
    solver: str | None = None,
    z_lin: np.ndarray | None = None,
    obs: np.ndarray | None = None,
    obs_r: float = 0.0,
    use_position_jacobian_for_obstacles: bool = True,
) -> np.ndarray:
    """Solve the SCP trust-region QP with CVXPY.

    The QP always contains the trust-region constraint and always calls the
    obstacle-constraint builder.  If the obstacle array is empty, no obstacle
    half-spaces are added, but the same constrained CVXPY solve is still used.
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
        raise ValueError("z_lin must be provided for the constrained conditioning problem.")

    dim = Jp.shape[1]
    eta = cp.Variable(dim)
    objective_expr = (
        cp.quad_form(rp + Jp @ eta, cp.psd_wrap(Pinv))
        + cp.quad_form(ry - H @ eta, cp.psd_wrap(Ninv))
    )

    installed = set(cp.installed_solvers())
    errors: list[str] = []

    # Always solve the constrained form: trust region plus the linearized
    # obstacle half-spaces.  This removes the old unconstrained shortcut so the
    # state-conditioning step always has the same constraint structure.
    constraints = [cp.sum_squares(eta) <= float(trust_radius) ** 2]

    # H maps intrinsic eta to first-order GPS/position change because
    # y - h(z Exp(eta)) ≈ ry - H eta.  Thus h(z Exp(eta)) ≈ h(z) + H eta.
    H_p = H if use_position_jacobian_for_obstacles else None
    constraints += _obstacle_constraints(
        eta=eta,
        z_lin=np.asarray(z_lin, dtype=float).reshape(3),
        obs=obs_arr,
        obs_r=float(obs_r),
        position_jacobian=H_p,
    )

    constrained = cp.Problem(cp.Minimize(objective_expr), constraints)

    conic_candidates = []
    if solver is not None:
        conic_candidates.append(solver)
    conic_candidates += ["CLARABEL", "MOSEK", "SCS", "SCIPY", "OSQP", "HIGHS"]

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


def conditioning_objective_SE2(
    z_pred: np.ndarray,
    z: np.ndarray,
    y_gps: np.ndarray,
    Pinv: np.ndarray,
    Ninv: np.ndarray,
) -> float:
    """Evaluate the nonlinear conditioning objective at a state estimate.

    This is the objective used for the SCP stopping condition between two
    successive iterates:

        ||Log(z_pred^{-1} z)||_{P_pred^{-1}}^2
        + ||y_gps - h(z)||_{N^{-1}}^2.
    """
    rp = inv_retract_SE2(np.asarray(z_pred, dtype=float).reshape(3),
                         np.asarray(z, dtype=float).reshape(3))
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
    trust_radius: float = 0.5,
    max_scp_iters: int = 10,
    tol: float = 1e-9,
    solver: str | None = None,
    fallback_without_cvxpy: bool = False,
    obs: np.ndarray | None = None,
    obs_r: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Successive convexified intrinsic GPS update on SE(2), CVXPY version.

    Returns
    -------
    z_upd, P_upd, used_cvxpy
        ``used_cvxpy`` is always True for this constrained CVXPY version.
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
            "as a constrained CVXPY problem. Install it with `pip install cvxpy`."
        )
    used_cvxpy = True

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
        # Therefore the first-order position perturbation is H eta.
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
            z_lin=z_iter,
            obs=obs_arr,
            obs_r=obs_r,
            use_position_jacobian_for_obstacles=True,
        )

        z_iter = retract_SE2(z_iter, step)
        curr_obj = conditioning_objective_SE2(
            z_pred=z_pred,
            z=z_iter,
            y_gps=y_gps,
            Pinv=Pinv,
            Ninv=Ninv,
        )
        obj_diff = abs(curr_obj - prev_obj)
        obj_scale = 1.0 + abs(prev_obj)
        if obj_diff <= float(tol) * obj_scale:
            break
        prev_obj = curr_obj

    z_upd = z_iter.copy()

    # Gauss-Newton posterior covariance at the final iterate.
    Jp = prior_residual_jacobian_SE2(z_pred, z_upd)
    H = gps_measurement_jacobian_intrinsic(z_upd)
    Hess = Jp.T @ Pinv @ Jp + H.T @ Ninv @ H
    Hess = 0.5 * (Hess + Hess.T)
    P_upd = np.linalg.inv(Hess + 1e-12 * np.eye(3))
    P_upd = 0.5 * (P_upd + P_upd.T)

    return z_upd, P_upd, used_cvxpy
