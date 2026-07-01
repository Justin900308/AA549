"""Continuous-discrete LIEKF prediction with fixed-chart SCP conditioning.

Prediction is unchanged.  At a measurement time, the predicted state chi_pred
defines one fixed Lie-algebra chart

    chi(delta) = chi_pred Exp(delta^).

The exact full homogeneous conditioning problem is

    min_delta ||delta||_{P_pred^{-1}}^2
              + || z - (Exp(delta^) - I)d ||_{N_hat^{-1}}^2

    s.t.      r_obs^2 - ||p(chi_pred Exp(delta^)) - p_obs,j||^2 <= 0,

where z = chi_pred^{-1}Y-d and
N_hat^{-1}=chi_pred^T N^{-1} chi_pred.

This implementation performs ordinary SCP directly on the fixed Euclidean
coordinate delta.  At iteration i it uses an additive decision step s:

    delta = delta_i + s,

and solves

    min_s ||delta_i+s||_{P_pred^{-1}}^2
          + ||z-phi(delta_i)-H_i s||_{N_hat^{-1}}^2

subject to ||s||_2 <= trust_radius and linearized obstacle constraints.
No intermediate group state chi_i is maintained by the SCP loop.  A pose is
formed only as the derived value chi_pred Exp(delta_i^) when needed to evaluate
physical obstacle functions, and once at the end to return the corrected state.
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CVXPY_AVAILABLE = False

from dynamics import (
    SE2_GPS_D,
    SE2_to_pose,
    as_full_homogeneous_covariance_SE2,
    as_full_homogeneous_measurement_SE2,
    full_left_invariant_information_SE2,
    full_left_invariant_innovation_SE2,
    full_left_invariant_measurement_jacobian_from_delta_SE2,
    full_left_invariant_measurement_model_from_delta_SE2,
    full_left_invariant_measurement_jacobian_SE2,
    full_prior_residual_jacobian_SE2,
    full_world_position_from_delta_SE2,
    full_world_position_jacobian_from_delta_SE2,
    liekf_A_matrix,
    pose_to_SE2,
    se2_exp,
    unicycle_dynamics,
    wrap_angle,
)
from integrator import covariance_euler, rk4


class ContinuousDiscreteCarISCVXCVXPY:
    """LIEKF prediction plus full homogeneous-vector ISCVX conditioning.

    Existing use remains valid:

        filt = ContinuousDiscreteCarISCVXCVXPY(..., N=N_xy, ...)
        filt.update(y_gps=[x_meas, y_meas])

    In this case the code internally embeds the 2D GPS observation into the
    full 3-vector/matrix derivation.  Advanced use may pass a full 3-vector
    observation and an SPD 3x3 covariance directly.

    Parameters
    ----------
    measurement_d:
        Full known vector d in Y=chi d+V.  Defaults to [0,0,1]^T for GPS.
    homogeneous_variance:
        Positive dummy variance used only when ``N`` is supplied as 2x2.
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
        measurement_d: np.ndarray | None = None,
        homogeneous_variance: float = 1.0,
    ) -> None:
        self.z = np.asarray(z0, dtype=float).reshape(3).copy()
        self.z[0] = wrap_angle(self.z[0])
        self.P = np.asarray(P0, dtype=float).reshape(3, 3).copy()
        self.Q = np.asarray(Q, dtype=float).reshape(3, 3)
        self.N = np.asarray(N, dtype=float).copy()
        if self.N.shape not in {(2, 2), (3, 3)}:
            raise ValueError("N must have shape (2,2) or (3,3).")
        self.dt = float(dt)
        self.trust_radius = float(trust_radius)
        self.max_scp_iters = int(max_scp_iters)
        self.tol = float(tol)
        self.solver = solver
        self.fallback_without_cvxpy = bool(fallback_without_cvxpy)
        self.obs = _format_obstacles(obs)
        self.obs_r = float(obs_r)
        self.measurement_d = (
            SE2_GPS_D.copy()
            if measurement_d is None
            else np.asarray(measurement_d, dtype=float).reshape(3).copy()
        )
        self.homogeneous_variance = float(homogeneous_variance)
        self.used_cvxpy = False
        self.used_fallback = False

        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required because the ISCVX conditioning step is solved "
                "as a constrained CVXPY problem. Install it with `pip install cvxpy`."
            )

    def predict(self, u) -> None:
        """Unchanged continuous-discrete LIEKF prediction."""
        u = np.asarray(u, dtype=float).reshape(2)
        v, omega = float(u[0]), float(u[1])
        A = liekf_A_matrix(v, omega)

        self.z = rk4(unicycle_dynamics, self.z, u, self.dt)
        self.P = covariance_euler(self.P, A, self.Q, self.dt)
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-15 * np.eye(3)

    def update(self, Y_or_y) -> None:
        self.z, self.P, used_cvxpy = intrinsic_cvxpy_update_SE2(
            z_pred=self.z,
            P_pred=self.P,
            Y_or_y=Y_or_y,
            N=self.N,
            d=self.measurement_d,
            homogeneous_variance=self.homogeneous_variance,
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

    def step(self, u, Y_or_y=None) -> np.ndarray:
        self.predict(u)
        if Y_or_y is not None:
            self.update(Y_or_y)
        return self.z.copy()


# -----------------------------------------------------------------------------
# Numeric and obstacle helpers
# -----------------------------------------------------------------------------


def _format_obstacles(obs: np.ndarray | None) -> np.ndarray:
    """Return obstacle centers with shape (num_obstacles,2)."""
    if obs is None:
        return np.zeros((0, 2), dtype=float)
    obs_arr = np.asarray(obs, dtype=float)
    if obs_arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    return obs_arr.reshape(-1, 2)


def _project_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize and project a matrix to SPD for a CVXPY quadratic form."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    A_spd = (vecs * vals) @ vecs.T
    return 0.5 * (A_spd + A_spd.T)


def _safe_inverse_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Numerically safe inverse of an SPD matrix."""
    A = _project_spd(A, eps=eps)
    vals, vecs = np.linalg.eigh(A)
    Ainv = (vecs * (1.0 / vals)) @ vecs.T
    return 0.5 * (Ainv + Ainv.T)


def _obstacle_constraints(
    eta,
    p_iter: np.ndarray,
    obs: np.ndarray,
    obs_r: float,
    H_p_world: np.ndarray,
) -> list:
    """Linearized physical circular keep-out constraints.

    The candidate state in each SCP subproblem is chi_iter Exp(eta^).  Its
    world-frame position perturbation is H_p_world eta, even though the
    measurement residual itself is represented by full invariant vectors.
    """
    p_iter = np.asarray(p_iter, dtype=float).reshape(2)
    H_p_world = np.asarray(H_p_world, dtype=float).reshape(2, -1)
    constraints = []
    for obs_j in _format_obstacles(obs):
        diff = p_iter - obs_j
        h_j = float(obs_r) ** 2 - diff @ diff
        a_j = -2.0 * diff
        constraints.append(h_j + a_j @ (H_p_world @ eta) <= 0.0)
    return constraints


# -----------------------------------------------------------------------------
# CVXPY SCP subproblem
# -----------------------------------------------------------------------------


def solve_trust_region_qp_cvxpy(
    delta_i: np.ndarray,
    Pinv: np.ndarray,
    H: np.ndarray,
    Nhatinv: np.ndarray,
    ry: np.ndarray,
    trust_radius: float,
    *,
    p_iter: np.ndarray,
    H_p_world: np.ndarray,
    obs: np.ndarray | None = None,
    obs_r: float = 0.0,
    solver: str | None = None,
) -> np.ndarray:
    """Solve one fixed-chart SCP subproblem for the additive step ``s``.

    The nonlinear problem is parameterized by the fixed coordinate delta.
    At iteration i, delta = delta_i + s and the subproblem is

        min_s ||delta_i+s||_{Pinv}^2
              + ||ry-Hs||_{Nhatinv}^2

    with the usual trust-region and linearized obstacle constraints.
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("cvxpy is not installed. Install it with `pip install cvxpy`.")

    delta_i = np.asarray(delta_i, dtype=float).reshape(3)
    H = np.asarray(H, dtype=float)
    ry = np.asarray(ry, dtype=float).reshape(-1)
    Pinv = _project_spd(Pinv)
    Nhatinv = _project_spd(Nhatinv)

    step = cp.Variable(3)
    objective = (
        cp.quad_form(delta_i + step, cp.psd_wrap(Pinv))
        + cp.quad_form(ry - H @ step, cp.psd_wrap(Nhatinv))
    )
    constraints = [cp.sum_squares(step) <= float(trust_radius) ** 2]
    constraints += _obstacle_constraints(
        eta=step,
        p_iter=p_iter,
        obs=_format_obstacles(obs),
        obs_r=float(obs_r),
        H_p_world=H_p_world,
    )

    problem = cp.Problem(cp.Minimize(objective), constraints)
    candidates = []
    if solver is not None:
        candidates.append(solver)
    candidates += ["CLARABEL", "MOSEK", "SCS", "SCIPY", "OSQP", "HIGHS"]

    errors: list[str] = []
    installed = set(cp.installed_solvers())
    for name in candidates:
        if name not in installed:
            continue
        try:
            kwargs = {}
            if name == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 5000})
            problem.solve(solver=name, verbose=False, **kwargs)
            if step.value is not None and problem.status in {
                "optimal",
                "optimal_inaccurate",
            }:
                return np.asarray(step.value, dtype=float).reshape(3)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    message = "CVXPY failed to solve the fixed-chart SCP subproblem."
    if errors:
        message += " Solver errors: " + " | ".join(errors)
    raise RuntimeError(message)


# -----------------------------------------------------------------------------
# Exact full conditioning objective and ISCVX loop
# -----------------------------------------------------------------------------


def conditioning_objective_full_SE2(
    delta: np.ndarray,
    z_invariant: np.ndarray,
    d: np.ndarray,
    Pinv: np.ndarray,
    Nhatinv: np.ndarray,
) -> float:
    """Evaluate the exact nonlinear objective in the fixed delta chart.

        F(delta) = ||delta||_{Pinv}^2
                 + ||z - (Exp(delta^) - I)d||_{Nhatinv}^2.
    """
    delta = np.asarray(delta, dtype=float).reshape(3)
    z_invariant = np.asarray(z_invariant, dtype=float).reshape(3)
    d = np.asarray(d, dtype=float).reshape(3)
    phi = full_left_invariant_measurement_model_from_delta_SE2(delta, d)
    residual = z_invariant - phi
    return float(delta.T @ Pinv @ delta + residual.T @ Nhatinv @ residual)


def intrinsic_cvxpy_update_SE2(
    z_pred: np.ndarray,
    P_pred: np.ndarray,
    Y_or_y: np.ndarray,
    N: np.ndarray,
    *,
    d: np.ndarray | None = None,
    homogeneous_variance: float = 1.0,
    trust_radius: float = 0.5,
    max_scp_iters: int = 10,
    tol: float = 1e-9,
    solver: str | None = None,
    fallback_without_cvxpy: bool = False,
    obs: np.ndarray | None = None,
    obs_r: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve the full homogeneous conditioning problem by fixed-chart SCP.

    The solver works directly with

        delta = Log(chi_pred^{-1} chi)

    and uses the additive Euclidean update

        delta_{i+1} = delta_i + s_i.

    No group iterate is propagated in the SCP loop.  The final corrected pose
    is recovered once after convergence:

        chi_plus = chi_pred Exp(delta_star^).
    """
    del fallback_without_cvxpy  # retained in the signature for compatibility

    z_pred = np.asarray(z_pred, dtype=float).reshape(3)
    chi_pred = pose_to_SE2(z_pred)
    P_pred = np.asarray(P_pred, dtype=float).reshape(3, 3)
    d = SE2_GPS_D.copy() if d is None else np.asarray(d, dtype=float).reshape(3)
    Y = as_full_homogeneous_measurement_SE2(Y_or_y)
    N_full = as_full_homogeneous_covariance_SE2(N, homogeneous_variance)

    Pinv = _safe_inverse_spd(P_pred)
    Nhatinv = full_left_invariant_information_SE2(chi_pred, N_full)
    z_invariant = full_left_invariant_innovation_SE2(chi_pred, Y, d)

    if not CVXPY_AVAILABLE:
        raise ImportError(
            "cvxpy is required because the ISCVX conditioning step is solved "
            "as a constrained CVXPY problem. Install it with `pip install cvxpy`."
        )
    used_cvxpy = True

    # Fixed coordinate of the candidate state relative to chi_pred.
    delta_iter = np.zeros(3, dtype=float)
    prev_obj = conditioning_objective_full_SE2(
        delta=delta_iter,
        z_invariant=z_invariant,
        d=d,
        Pinv=Pinv,
        Nhatinv=Nhatinv,
    )

    for _ in range(int(max_scp_iters)):
        # Fixed-chart model:
        #
        #   phi(delta_iter + s) ~= phi(delta_iter) + H_i s,
        #
        # so z - phi(delta_iter+s) ~= ry - H_i s.  The prior is exact
        # in the SCP decision variable: ||delta_iter+s||_{P^{-1}}^2.
        phi_i = full_left_invariant_measurement_model_from_delta_SE2(
            delta_iter, d
        )
        ry = z_invariant - phi_i
        H_i = full_left_invariant_measurement_jacobian_from_delta_SE2(
            delta_iter, d
        )

        # Constraints are also functions of delta.  These two values merely
        # evaluate p(chi_pred Exp(delta_iter^)); no chi_i is maintained.
        p_iter = full_world_position_from_delta_SE2(chi_pred, delta_iter)
        H_p_world = full_world_position_jacobian_from_delta_SE2(
            chi_pred, delta_iter
        )

        step = solve_trust_region_qp_cvxpy(
            delta_i=delta_iter,
            Pinv=Pinv,
            H=H_i,
            Nhatinv=Nhatinv,
            ry=ry,
            trust_radius=trust_radius,
            p_iter=p_iter,
            H_p_world=H_p_world,
            obs=obs,
            obs_r=obs_r,
            solver=solver,
        )

        delta_next = delta_iter + step
        current_obj = conditioning_objective_full_SE2(
            delta=delta_next,
            z_invariant=z_invariant,
            d=d,
            Pinv=Pinv,
            Nhatinv=Nhatinv,
        )

        # Preserve the original objective-difference stopping logic.
        delta_iter = delta_next
        if abs(current_obj - prev_obj) <= float(tol) * (1.0 + abs(prev_obj)):
            break
        prev_obj = current_obj

    delta_star = delta_iter
    chi_upd = chi_pred @ se2_exp(delta_star)

    # Preserve the original covariance convention: form the Gauss--Newton
    # Hessian in the *final local right-retraction coordinate* eta about
    # chi_upd.  This is the covariance coordinate expected by the next filter
    # cycle, not the fixed predicted-chart delta coordinate.
    Jp_final = full_prior_residual_jacobian_SE2(chi_pred, chi_upd)
    H_final = full_left_invariant_measurement_jacobian_SE2(
        chi_pred, chi_upd, d
    )
    Hess = Jp_final.T @ Pinv @ Jp_final + H_final.T @ Nhatinv @ H_final
    Hess = 0.5 * (Hess + Hess.T)
    P_upd = np.linalg.inv(Hess + 1e-12 * np.eye(3))
    P_upd = 0.5 * (P_upd + P_upd.T)

    return SE2_to_pose(chi_upd), P_upd, used_cvxpy
