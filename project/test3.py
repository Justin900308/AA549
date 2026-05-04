import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# ============================================================
# Quaternion utilities: q = [w, x, y, z]
# ============================================================

def q_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def q_conj(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_mul(q, p):
    """Hamilton product q ⊗ p."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array([
        w1*w2 - x1*w2*0 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def q_inv(q):
    return q_conj(q) / np.dot(q, q)


def q_exp(delta):
    """
    Exp(delta) = [cos(||delta||), sin(||delta||)/||delta|| delta].
    delta is the half-angle vector.
    """
    delta = np.asarray(delta, dtype=float)
    th = np.linalg.norm(delta)
    if th < 1e-12:
        return q_normalize(np.hstack([1.0, delta]))
    return np.hstack([np.cos(th), np.sin(th) / th * delta])


def q_log(q):
    """
    Log(q) from S^3 to R^3.
    Returns half-angle vector.
    """
    q = q_normalize(q)

    # q and -q represent the same attitude; choose short branch.
    if q[0] < 0:
        q = -q

    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    nv = np.linalg.norm(v)

    if nv < 1e-12:
        return v.copy()

    th = np.arctan2(nv, w)
    return th * v / nv


def retract(q, delta):
    """R_q(delta) = q ⊗ Exp(delta)."""
    return q_normalize(q_mul(q, q_exp(delta)))


def inv_retract(q_base, q):
    """R^{-1}_{q_base}(q) = Log(q_base^{-1} ⊗ q)."""
    return q_log(q_mul(q_inv(q_base), q))


def quat_to_rot(q):
    """R(q): body vector -> inertial vector."""
    q = q_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_from_two_vectors(a, b):
    """
    Quaternion q such that R(q) a = b.
    Both a and b are 3D vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = np.clip(np.dot(a, b), -1.0, 1.0)

    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    if dot < -1.0 + 1e-12:
        # Pick any axis orthogonal to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.hstack([0.0, axis])

    axis = np.cross(a, b)
    q = np.hstack([1.0 + dot, axis])
    return q_normalize(q)


# ============================================================
# Dynamics and measurement
# ============================================================

def dynamics(q, omega, dt):
    """
    Quaternion kinematics:
        q_{k+1} = q_k ⊗ Exp(0.5 dt omega)
    """
    Omega = 0.5 * dt * omega
    return retract(q, Omega)


def h_measure(q, refs):
    """
    Vector observation model:
        y = [R(q)^T r_1; R(q)^T r_2; ...] + noise.
    """
    Rq = quat_to_rot(q)
    return np.concatenate([Rq.T @ r for r in refs])


# ============================================================
# Cone constraint
# ============================================================

def cone_margin(q, cone_axis, body_axis, alpha):
    """
    g(q) = cone_axis^T R(q) body_axis - cos(alpha).
    Feasible iff g(q) >= 0.
    """
    cone_axis = cone_axis / np.linalg.norm(cone_axis)
    body_axis = body_axis / np.linalg.norm(body_axis)
    d = quat_to_rot(q) @ body_axis
    return float(cone_axis @ d - np.cos(alpha))


def cone_angle(q, cone_axis, body_axis):
    """
    angle between R(q) body_axis and cone_axis.
    """
    cone_axis = cone_axis / np.linalg.norm(cone_axis)
    body_axis = body_axis / np.linalg.norm(body_axis)
    d = quat_to_rot(q) @ body_axis
    val = np.clip(cone_axis @ d, -1.0, 1.0)
    return float(np.arccos(val))


def cone_violation_angle(q, cone_axis, body_axis, alpha):
    """
    positive value means violation.
    """
    return max(0.0, cone_angle(q, cone_axis, body_axis) - alpha)


def project_quat_to_cone(q, cone_axis, body_axis, alpha):
    """
    Project only the constrained direction R(q) body_axis onto the cone.
    The remaining yaw/twist is preserved approximately by left-multiplying
    a minimal inertial rotation.
    """
    q = q_normalize(q)

    if cone_margin(q, cone_axis, body_axis, alpha) >= 0:
        return q

    c = cone_axis / np.linalg.norm(cone_axis)
    b = body_axis / np.linalg.norm(body_axis)

    d = quat_to_rot(q) @ b
    d = d / np.linalg.norm(d)

    # Direction from c toward d.
    u = d - (c @ d) * c
    nu = np.linalg.norm(u)

    if nu < 1e-10:
        # Degenerate case: choose arbitrary perpendicular direction.
        u = np.cross(c, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(u) < 1e-10:
            u = np.cross(c, np.array([0.0, 1.0, 0.0]))
        u = u / np.linalg.norm(u)
    else:
        u = u / nu

    d_proj = np.cos(alpha) * c + np.sin(alpha) * u
    d_proj = d_proj / np.linalg.norm(d_proj)

    q_corr = quat_from_two_vectors(d, d_proj)

    # Left multiply: R(q_corr ⊗ q) = R(q_corr) R(q)
    return q_normalize(q_mul(q_corr, q))


def cone_jacobian_intrinsic(q, cone_axis, body_axis, alpha):
    """
    G = d/deta g(R_q(eta)) at eta = 0.
    Returns shape (3,).
    """
    def g_pert(eta):
        return np.array([
            cone_margin(
                retract(q, eta),
                cone_axis=cone_axis,
                body_axis=body_axis,
                alpha=alpha,
            )
        ])

    return numerical_jacobian(g_pert, xdim=3).reshape(3)


# ============================================================
# Numerical Jacobians
# ============================================================

def numerical_jacobian(fun, xdim, eps=1e-6):
    f0 = fun(np.zeros(xdim))
    J = np.zeros((len(f0), xdim))

    for i in range(xdim):
        e = np.zeros(xdim)
        e[i] = eps
        fp = fun(e)
        fm = fun(-e)
        J[:, i] = (fp - fm) / (2 * eps)

    return J


def dynamics_error_jacobian_intrinsic(q_hat, omega, dt):
    """
    A = d/de Log(q_pred^{-1} ⊗ f(R_{q_hat}(e), omega)) at e=0.
    """
    q_pred = dynamics(q_hat, omega, dt)

    def err_next(e):
        q_pert = retract(q_hat, e)
        q_next = dynamics(q_pert, omega, dt)
        return inv_retract(q_pred, q_next)

    return numerical_jacobian(err_next, xdim=3)


def measurement_jacobian_intrinsic(q, refs):
    """
    H = d/de h(R_q(e)) at e=0.
    """
    def meas_pert(e):
        return h_measure(retract(q, e), refs)

    return numerical_jacobian(meas_pert, xdim=3)


def prior_residual_jacobian(q_pred, q_iter):
    """
    Jp = d/deta Log(q_pred^{-1} ⊗ R_{q_iter}(eta)) at eta=0.
    """
    def prior_pert(eta):
        return inv_retract(q_pred, retract(q_iter, eta))

    return numerical_jacobian(prior_pert, xdim=3)


def dynamics_jacobian_additive(q_hat, omega, dt):
    """
    A = d/dq f(q, omega) at q_hat.
    """
    def f_pert(dq):
        return dynamics(q_hat + dq, omega, dt)

    return numerical_jacobian(f_pert, xdim=4)


def measurement_jacobian_additive(q_hat, refs):
    """
    H = d/dq h(q) at q_hat.
    """
    def h_pert(dq):
        return h_measure(q_normalize(q_hat + dq), refs)

    return numerical_jacobian(h_pert, xdim=4)


# ============================================================
# Constrained intrinsic CVXPY/SCP update
# ============================================================

def intrinsic_cvx_update_constrained(
    q_pred,
    P_pred,
    y,
    R_meas,
    refs,
    cone_axis,
    body_axis,
    alpha,
    trust_radius=0.15,
    max_scp_iters=8,
    tol=1e-8,
):
    """
    Constrained ISCVX measurement update.

    At SCP iteration j:
        q = R_{q_j}(eta)

        prior:
            R^{-1}_{q_pred}(q) approx rp + Jp eta

        measurement:
            y - h(q) approx ry - H eta

        cone:
            g(q) = c^T R(q)b - cos(alpha) >= 0
            g(R_{q_j}(eta)) approx g_j + G_j eta >= 0

        solve:
            min ||rp + Jp eta||_{P^{-1}}^2
              + ||ry - H eta||_{R^{-1}}^2
            s.t. g_j + G_j eta >= 0
                 ||eta|| <= trust_radius
    """
    # Start from a feasible point.
    q_iter = project_quat_to_cone(q_pred, cone_axis, body_axis, alpha)

    Pinv = np.linalg.inv(P_pred)
    Rinv = np.linalg.inv(R_meas)

    Pinv = 0.5 * (Pinv + Pinv.T)
    Rinv = 0.5 * (Rinv + Rinv.T)

    for _ in range(max_scp_iters):
        rp = inv_retract(q_pred, q_iter)
        Jp = prior_residual_jacobian(q_pred, q_iter)

        ry = y - h_measure(q_iter, refs)
        H = measurement_jacobian_intrinsic(q_iter, refs)

        gj = cone_margin(q_iter, cone_axis, body_axis, alpha)
        Gj = cone_jacobian_intrinsic(q_iter, cone_axis, body_axis, alpha)

        eta = cp.Variable(3)

        objective = (
            cp.quad_form(rp + Jp @ eta, Pinv)
            + cp.quad_form(ry - H @ eta, Rinv)
        )

        constraints = [
            Gj @ eta + gj >= 0.0,
            cp.sum_squares(eta) <= trust_radius**2,
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=5000)

        if eta.value is None:
            raise RuntimeError("CVXPY failed to solve the constrained intrinsic update.")

        step = np.asarray(eta.value).reshape(3)

        # Feasibility safeguard for the true nonlinear cone constraint.
        # The SCP constraint is linearized, so we backtrack if needed.
        step_scale = 1.0
        q_candidate = retract(q_iter, step)

        while cone_margin(q_candidate, cone_axis, body_axis, alpha) < -1e-10 and step_scale > 1e-4:
            step_scale *= 0.5
            q_candidate = retract(q_iter, step_scale * step)

        q_iter = q_candidate

        # Final exact projection safeguard.
        q_iter = project_quat_to_cone(q_iter, cone_axis, body_axis, alpha)

        if np.linalg.norm(step_scale * step) < tol:
            break

    q_upd = q_iter

    # Gauss-Newton covariance at final point.
    Jp = prior_residual_jacobian(q_pred, q_upd)
    H = measurement_jacobian_intrinsic(q_upd, refs)

    Hess = Jp.T @ Pinv @ Jp + H.T @ Rinv @ H
    Hess = 0.5 * (Hess + Hess.T)

    P_upd = np.linalg.inv(Hess + 1e-12 * np.eye(3))
    P_upd = 0.5 * (P_upd + P_upd.T)

    return q_upd, P_upd


# ============================================================
# Intrinsic constrained quaternion filter
# ============================================================

def intrinsic_quaternion_filter_constrained(
    y_all,
    omega_all,
    refs,
    dt,
    q0_hat,
    P0,
    Q,
    R_meas,
    cone_axis,
    body_axis,
    alpha,
    trust_radius=0.15,
    max_scp_iters=8,
):
    N = len(omega_all)

    q_hat_all = np.zeros((N + 1, 4))
    P_all = np.zeros((N + 1, 3, 3))

    q_hat_all[0] = project_quat_to_cone(
        q_normalize(q0_hat),
        cone_axis,
        body_axis,
        alpha,
    )
    P_all[0] = P0

    for k in range(N):
        q_hat = q_hat_all[k]
        P = P_all[k]
        omega = omega_all[k]

        # Prediction
        q_pred = dynamics(q_hat, omega, dt)
        A = dynamics_error_jacobian_intrinsic(q_hat, omega, dt)

        P_pred = A @ P @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        # Constrained ISCVX update
        q_upd, P_upd = intrinsic_cvx_update_constrained(
            q_pred=q_pred,
            P_pred=P_pred,
            y=y_all[k + 1],
            R_meas=R_meas,
            refs=refs,
            cone_axis=cone_axis,
            body_axis=body_axis,
            alpha=alpha,
            trust_radius=trust_radius,
            max_scp_iters=max_scp_iters,
        )

        q_hat_all[k + 1] = q_upd
        P_all[k + 1] = P_upd

    return q_hat_all, P_all


# ============================================================
# Normal additive quaternion EKF
# ============================================================

def additive_quaternion_ekf(
    y_all,
    omega_all,
    refs,
    dt,
    q0_hat,
    P0_4d,
    Q4,
    R_meas,
):
    """
    Standard ambient quaternion EKF:
        q_pred = f(q)
        P_pred = A P A^T + Q
        dq = K(y - h(q_pred))
        q_upd = normalize(q_pred + dq)

    This enforces q in S^3 only. It does not enforce the cone.
    """
    N = len(omega_all)

    q_hat_all = np.zeros((N + 1, 4))
    P_all = np.zeros((N + 1, 4, 4))

    q_hat_all[0] = q_normalize(q0_hat)
    P_all[0] = P0_4d

    I4 = np.eye(4)

    for k in range(N):
        q_hat = q_hat_all[k]
        P = P_all[k]
        omega = omega_all[k]

        # Prediction
        q_pred = dynamics(q_hat, omega, dt)
        A = dynamics_jacobian_additive(q_hat, omega, dt)

        P_pred = A @ P @ A.T + Q4
        P_pred = 0.5 * (P_pred + P_pred.T)

        # Measurement update
        y_pred = h_measure(q_pred, refs)
        innov = y_all[k + 1] - y_pred

        H = measurement_jacobian_additive(q_pred, refs)

        S = H @ P_pred @ H.T + R_meas
        K = P_pred @ H.T @ np.linalg.inv(S)

        dq = K @ innov

        q_raw = q_pred + dq
        q_upd = q_normalize(q_raw)

        P_upd = (I4 - K @ H) @ P_pred @ (I4 - K @ H).T + K @ R_meas @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        q_hat_all[k + 1] = q_upd
        P_all[k + 1] = P_upd

    return q_hat_all, P_all


# ============================================================
# Simulation inside a cone
# ============================================================

def euler_to_quat(roll, pitch, yaw):
    """
    ZYX convention:
        q = q_z(yaw) ⊗ q_y(pitch) ⊗ q_x(roll)
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    return q_normalize(np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        sy*cp*sr + cy*sp*cr,
        sy*cp*cr - cy*sp*sr,
    ]))


def generate_true_cone_trajectory(N, dt, alpha, cone_axis, body_axis):
    """
    Generate a true trajectory whose body_axis remains inside the cone.
    We use small roll/pitch oscillations with amplitude below alpha.
    """
    q_true_all = np.zeros((N + 1, 4))

    # Keep safely inside the cone.
    amp = 0.55 * alpha

    for k in range(N + 1):
        t = k * dt

        roll = amp * np.sin(1.3 * t)
        pitch = 0.8 * amp * np.cos(1.1 * t)
        yaw = 0.6 * np.sin(0.5 * t)

        q = euler_to_quat(roll, pitch, yaw)

        # Safety projection in case Euler combination gets too close.
        q = project_quat_to_cone(q, cone_axis, body_axis, 0.95 * alpha)

        q_true_all[k] = q

    omega_true_all = np.zeros((N, 3))

    for k in range(N):
        dq = inv_retract(q_true_all[k], q_true_all[k + 1])
        omega_true_all[k] = (2.0 / dt) * dq

    return q_true_all, omega_true_all


def simulate_trial_cone(
    rng,
    N,
    dt,
    refs,
    gyro_sigma,
    meas_sigma,
    alpha,
    cone_axis,
    body_axis,
    outlier_strength=0.16,
):
    """
    Simulate true attitude inside cone.
    Add measurement noise and occasional outliers to make the unconstrained EKF
    visibly violate the cone.
    """
    meas_dim = 3 * len(refs)

    q_true_all, omega_true_all = generate_true_cone_trajectory(
        N=N,
        dt=dt,
        alpha=alpha,
        cone_axis=cone_axis,
        body_axis=body_axis,
    )

    omega_meas_all = omega_true_all + gyro_sigma * rng.standard_normal((N, 3))

    y_all = np.zeros((N + 1, meas_dim))

    for k in range(N + 1):
        y_clean = h_measure(q_true_all[k], refs)
        noise = meas_sigma * rng.standard_normal(meas_dim)

        # A few deliberate measurement outliers.
        # This is the easiest way to demonstrate why constraints matter:
        # the true state is feasible, but the noisy measurement suggests
        # a correction outside the feasible cone.
        if k in [12, 13, 24, 25, 36]:
            noise += outlier_strength * rng.standard_normal(meas_dim)

        y_all[k] = y_clean + noise

    return q_true_all, omega_meas_all, y_all


# ============================================================
# Error metrics
# ============================================================

def attitude_error_angle(q_hat, q_true):
    """
    Full attitude angle error in radians.
    Since Log returns half-angle vector, angle = 2||Log(q_hat^{-1} q_true)||.
    """
    e = inv_retract(q_hat, q_true)
    return 2.0 * np.linalg.norm(e)


# ============================================================
# Monte Carlo comparison
# ============================================================

def run_constrained_comparison(num_trials=10):
    base_rng = np.random.default_rng(7)

    dt = 0.05
    N = 50

    # Cone: body z-axis must remain within alpha of inertial z-axis.
    cone_axis = np.array([0.0, 0.0, 1.0])
    body_axis = np.array([0.0, 0.0, 1.0])
    alpha = np.deg2rad(12.0)

    refs = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.2, 0.1],
    ])
    refs = refs / np.linalg.norm(refs, axis=1, keepdims=True)

    meas_dim = 3 * len(refs)

    gyro_sigma = 0.015
    meas_sigma = 0.445

    R_meas = (meas_sigma**2) * np.eye(meas_dim)

    # Intrinsic tangent covariance
    P0_3d = (0.22**2) * np.eye(3)
    Q3 = (0.5 * dt * gyro_sigma)**2 * np.eye(3)

    # Additive ambient covariance
    P0_4d = (0.22**2) * np.eye(4)
    Q4 = (0.5 * dt * gyro_sigma)**2 * np.eye(4)

    err_iscvx_all = np.zeros((num_trials, N + 1))
    err_add_all = np.zeros((num_trials, N + 1))

    viol_iscvx_all = np.zeros((num_trials, N + 1))
    viol_add_all = np.zeros((num_trials, N + 1))

    true_angle_all = np.zeros((num_trials, N + 1))
    iscvx_angle_all = np.zeros((num_trials, N + 1))
    add_angle_all = np.zeros((num_trials, N + 1))

    for trial in range(num_trials):
        rng = np.random.default_rng(base_rng.integers(0, 10_000_000))

        q_true_all, omega_meas_all, y_all = simulate_trial_cone(
            rng=rng,
            N=N,
            dt=dt,
            refs=refs,
            gyro_sigma=gyro_sigma,
            meas_sigma=meas_sigma,
            alpha=alpha,
            cone_axis=cone_axis,
            body_axis=body_axis,
            outlier_strength=0.17,
        )

        # Same feasible initial estimate for both filters.
        init_error = np.array([0.06, -0.05, 0.04]) + 0.02 * rng.standard_normal(3)
        q0_hat = retract(q_true_all[0], init_error)
        q0_hat = project_quat_to_cone(q0_hat, cone_axis, body_axis, 0.90 * alpha)

        q_iscvx_all, P_iscvx_all = intrinsic_quaternion_filter_constrained(
            y_all=y_all,
            omega_all=omega_meas_all,
            refs=refs,
            dt=dt,
            q0_hat=q0_hat,
            P0=P0_3d,
            Q=Q3,
            R_meas=R_meas,
            cone_axis=cone_axis,
            body_axis=body_axis,
            alpha=alpha,
            trust_radius=0.12,
            max_scp_iters=8,
        )

        q_add_all, P_add_all = additive_quaternion_ekf(
            y_all=y_all,
            omega_all=omega_meas_all,
            refs=refs,
            dt=dt,
            q0_hat=q0_hat,
            P0_4d=P0_4d,
            Q4=Q4,
            R_meas=R_meas,
        )

        for k in range(N + 1):
            err_iscvx_all[trial, k] = attitude_error_angle(q_iscvx_all[k], q_true_all[k])
            err_add_all[trial, k] = attitude_error_angle(q_add_all[k], q_true_all[k])

            viol_iscvx_all[trial, k] = cone_violation_angle(
                q_iscvx_all[k], cone_axis, body_axis, alpha
            )
            viol_add_all[trial, k] = cone_violation_angle(
                q_add_all[k], cone_axis, body_axis, alpha
            )

            true_angle_all[trial, k] = cone_angle(q_true_all[k], cone_axis, body_axis)
            iscvx_angle_all[trial, k] = cone_angle(q_iscvx_all[k], cone_axis, body_axis)
            add_angle_all[trial, k] = cone_angle(q_add_all[k], cone_axis, body_axis)

        print(
            f"Trial {trial + 1:02d}: "
            f"max cone violation ISCVX = {np.rad2deg(np.max(viol_iscvx_all[trial])):.4f} deg, "
            f"additive EKF = {np.rad2deg(np.max(viol_add_all[trial])):.4f} deg"
        )

    time = np.arange(N + 1) * dt

    rms_iscvx = np.sqrt(np.mean(err_iscvx_all**2, axis=1))
    rms_add = np.sqrt(np.mean(err_add_all**2, axis=1))

    max_viol_iscvx = np.max(viol_iscvx_all, axis=1)
    max_viol_add = np.max(viol_add_all, axis=1)

    print("\n========== Summary ==========")
    print(f"ISCVX mean trajectory RMSE:     {np.mean(rms_iscvx):.6f} rad")
    print(f"Additive EKF mean trajectory RMSE: {np.mean(rms_add):.6f} rad")
    print(f"ISCVX max cone violation mean:  {np.rad2deg(np.mean(max_viol_iscvx)):.6f} deg")
    print(f"Additive max violation mean:    {np.rad2deg(np.mean(max_viol_add)):.6f} deg")
    print(f"ISCVX violated trials:          {np.sum(max_viol_iscvx > 1e-8)} / {num_trials}")
    print(f"Additive EKF violated trials:   {np.sum(max_viol_add > 1e-8)} / {num_trials}")

    # --------------------------------------------------------
    # Plot attitude error
    # --------------------------------------------------------
    mean_iscvx = np.mean(err_iscvx_all, axis=0)
    std_iscvx = np.std(err_iscvx_all, axis=0)

    mean_add = np.mean(err_add_all, axis=0)
    std_add = np.std(err_add_all, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(time, mean_iscvx, label="Constrained ISCVX-EKF")
    plt.fill_between(time, mean_iscvx - std_iscvx, mean_iscvx + std_iscvx, alpha=0.2)

    plt.plot(time, mean_add, label="Additive retraction EKF")
    plt.fill_between(time, mean_add - std_add, mean_add + std_add, alpha=0.2)

    plt.xlabel("time [s]")
    plt.ylabel("attitude error angle [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot cone angle
    # --------------------------------------------------------
    plt.figure(figsize=(8, 4))

    plt.plot(time, np.rad2deg(np.mean(true_angle_all, axis=0)), label="true cone angle")
    plt.plot(time, np.rad2deg(np.mean(iscvx_angle_all, axis=0)), label="ISCVX estimate")
    plt.plot(time, np.rad2deg(np.mean(add_angle_all, axis=0)), label="additive EKF estimate")

    plt.axhline(np.rad2deg(alpha), linestyle="--", label="cone boundary")

    plt.xlabel("time [s]")
    plt.ylabel("angle from cone axis [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot cone violation
    # --------------------------------------------------------
    plt.figure(figsize=(8, 4))

    plt.plot(
        time,
        np.rad2deg(np.max(viol_iscvx_all, axis=0)),
        label="max ISCVX violation over trials",
    )
    plt.plot(
        time,
        np.rad2deg(np.max(viol_add_all, axis=0)),
        label="max additive EKF violation over trials",
    )

    plt.xlabel("time [s]")
    plt.ylabel("cone violation [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Boxplot of max cone violation
    # --------------------------------------------------------
    plt.figure(figsize=(5, 4))
    plt.boxplot(
        [
            np.rad2deg(max_viol_iscvx),
            np.rad2deg(max_viol_add),
        ],
        labels=["Constrained\nISCVX-EKF", "Additive\nEKF"],
    )
    plt.ylabel("max cone violation [deg]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "time": time,
        "err_iscvx_all": err_iscvx_all,
        "err_add_all": err_add_all,
        "viol_iscvx_all": viol_iscvx_all,
        "viol_add_all": viol_add_all,
        "rms_iscvx": rms_iscvx,
        "rms_add": rms_add,
        "max_viol_iscvx": max_viol_iscvx,
        "max_viol_add": max_viol_add,
    }


if __name__ == "__main__":
    results = run_constrained_comparison(num_trials=10)