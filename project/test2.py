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
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
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
    3x3 tangent-space dynamics Jacobian.
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
    m x 3.
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
    4x4 ambient-space dynamics Jacobian.
    """
    def f_pert(dq):
        return dynamics(q_hat + dq, omega, dt)

    return numerical_jacobian(f_pert, xdim=4)


def measurement_jacobian_additive(q_hat, refs):
    """
    H = d/dq h(q) at q_hat.
    m x 4 ambient-space measurement Jacobian.
    """
    def h_pert(dq):
        return h_measure(q_normalize(q_hat + dq), refs)

    return numerical_jacobian(h_pert, xdim=4)


# ============================================================
# Intrinsic CVXPY/SCP measurement update
# ============================================================

def intrinsic_cvx_update(
    q_pred,
    P_pred,
    y,
    R_meas,
    refs,
    trust_radius=0.25,
    max_scp_iters=5,
    tol=1e-8,
):
    """
    Successive convexified intrinsic update.

    At SCP iteration j:
        q = R_{q_j}(eta)

        prior residual:
            R^{-1}_{q_pred}(q) ≈ rp + Jp eta

        measurement residual:
            y - h(q) ≈ ry - H eta

        solve:
            min ||rp + Jp eta||_{P^{-1}}^2
              + ||ry - H eta||_{R^{-1}}^2
            s.t. ||eta|| <= trust_radius
    """
    q_iter = q_pred.copy()

    Pinv = np.linalg.inv(P_pred)
    Rinv = np.linalg.inv(R_meas)

    Pinv = 0.5 * (Pinv + Pinv.T)
    Rinv = 0.5 * (Rinv + Rinv.T)

    for _ in range(max_scp_iters):
        rp = inv_retract(q_pred, q_iter)
        Jp = prior_residual_jacobian(q_pred, q_iter)

        ry = y - h_measure(q_iter, refs)
        H = measurement_jacobian_intrinsic(q_iter, refs)

        eta = cp.Variable(3)

        objective = (
            cp.quad_form(rp + Jp @ eta, Pinv)
            + cp.quad_form(ry - H @ eta, Rinv)
        )

        constraints = [
            cp.sum_squares(eta) <= trust_radius**2
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=5000)

        if eta.value is None:
            raise RuntimeError("CVXPY failed to solve the intrinsic update.")

        step = np.asarray(eta.value).reshape(3)
        q_iter = retract(q_iter, step)

        if np.linalg.norm(step) < tol:
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
# Intrinsic quaternion filter
# ============================================================

def intrinsic_quaternion_filter(
    y_all,
    omega_all,
    refs,
    dt,
    q0_hat,
    P0,
    Q,
    R_meas,
    trust_radius=0.25,
    max_scp_iters=5,
):
    N = len(omega_all)

    q_hat_all = np.zeros((N + 1, 4))
    P_all = np.zeros((N + 1, 3, 3))

    q_hat_all[0] = q_normalize(q0_hat)
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

        # Intrinsic CVXPY/SCP update
        q_upd, P_upd = intrinsic_cvx_update(
            q_pred=q_pred,
            P_pred=P_pred,
            y=y_all[k + 1],
            R_meas=R_meas,
            refs=refs,
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

        # Measurement linearization
        y_pred = h_measure(q_pred, refs)
        innov = y_all[k + 1] - y_pred

        H = measurement_jacobian_additive(q_pred, refs)

        S = H @ P_pred @ H.T + R_meas
        K = P_pred @ H.T @ np.linalg.inv(S)

        dq = K @ innov

        # Additive correction followed by normalization
        q_raw = q_pred + dq
        q_upd = q_normalize(q_raw)

        # Joseph covariance update
        P_upd = (I4 - K @ H) @ P_pred @ (I4 - K @ H).T + K @ R_meas @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        q_hat_all[k + 1] = q_upd
        P_all[k + 1] = P_upd

    return q_hat_all, P_all


# ============================================================
# Simulation and 10-trial comparison
# ============================================================

def simulate_trial(rng, N, dt, refs, gyro_sigma, meas_sigma):
    """
    Simulate true trajectory, noisy angular velocity input, and vector measurements.
    """
    meas_dim = 3 * len(refs)

    omega_true_all = np.zeros((N, 3))
    for k in range(N):
        t = k * dt
        omega_true_all[k] = np.array([
            0.4 * np.sin(0.7 * t),
            0.3 * np.cos(0.5 * t),
            0.2 + 0.1 * np.sin(0.3 * t),
        ])

    omega_meas_all = omega_true_all + gyro_sigma * rng.standard_normal((N, 3))

    q_true_all = np.zeros((N + 1, 4))
    q_true_all[0] = q_normalize(np.array([1.0, 0.0, 0.0, 0.0]))

    for k in range(N):
        q_true_all[k + 1] = dynamics(q_true_all[k], omega_true_all[k], dt)

    y_all = np.zeros((N + 1, meas_dim))
    for k in range(N + 1):
        y_all[k] = h_measure(q_true_all[k], refs) + meas_sigma * rng.standard_normal(meas_dim)

    return q_true_all, omega_meas_all, y_all


def attitude_error_angle(q_hat, q_true):
    """
    Full attitude angle error in radians.
    Since Log returns half-angle vector, angle ≈ 2||Log(q_hat^{-1} q_true)||.
    """
    e = inv_retract(q_hat, q_true)
    return 2.0 * np.linalg.norm(e)


def run_monte_carlo_comparison(num_trials=10):
    base_rng = np.random.default_rng(10)

    dt = 0.05
    N = 40

    refs = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.2, 0.1],
    ])
    refs = refs / np.linalg.norm(refs, axis=1, keepdims=True)

    meas_dim = 3 * len(refs)

    gyro_sigma = 0.015
    meas_sigma = 0.035

    R_meas = (meas_sigma**2) * np.eye(meas_dim)

    # Intrinsic tangent covariance
    P0_3d = (0.25**2) * np.eye(3)
    Q3 = (0.5 * dt * gyro_sigma)**2 * np.eye(3)

    # Additive ambient covariance
    P0_4d = (0.25**2) * np.eye(4)
    Q4 = (0.5 * dt * gyro_sigma)**2 * np.eye(4)

    err_intrinsic_all = np.zeros((num_trials, N + 1))
    err_additive_all = np.zeros((num_trials, N + 1))

    for trial in range(num_trials):
        rng = np.random.default_rng(base_rng.integers(0, 10_000_000))

        q_true_all, omega_meas_all, y_all = simulate_trial(
            rng=rng,
            N=N,
            dt=dt,
            refs=refs,
            gyro_sigma=gyro_sigma,
            meas_sigma=meas_sigma,
        )

        # Same initial estimate for both filters.
        init_error = np.array([0.18, -0.12, 0.10]) + 0.04 * rng.standard_normal(3)
        q0_hat = retract(q_true_all[0], init_error)

        q_intr_all, P_intr_all = intrinsic_quaternion_filter(
            y_all=y_all,
            omega_all=omega_meas_all,
            refs=refs,
            dt=dt,
            q0_hat=q0_hat,
            P0=P0_3d,
            Q=Q3,
            R_meas=R_meas,
            trust_radius=0.25,
            max_scp_iters=5,
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
            err_intrinsic_all[trial, k] = attitude_error_angle(q_intr_all[k], q_true_all[k])
            err_additive_all[trial, k] = attitude_error_angle(q_add_all[k], q_true_all[k])

        print(
            f"Trial {trial + 1:02d}: "
            f"final intrinsic = {err_intrinsic_all[trial, -1]:.4f} rad, "
            f"final additive = {err_additive_all[trial, -1]:.4f} rad"
        )

    time = np.arange(N + 1) * dt

    mean_intr = np.mean(err_intrinsic_all, axis=0)
    std_intr = np.std(err_intrinsic_all, axis=0)

    mean_add = np.mean(err_additive_all, axis=0)
    std_add = np.std(err_additive_all, axis=0)

    rms_intr = np.sqrt(np.mean(err_intrinsic_all**2, axis=1))
    rms_add = np.sqrt(np.mean(err_additive_all**2, axis=1))

    print("\n========== 10-Trial Summary ==========")
    print(f"Intrinsic CVXPY/SCP filter mean trajectory RMSE: {np.mean(rms_intr):.6f} rad")
    print(f"Additive quaternion EKF mean trajectory RMSE:   {np.mean(rms_add):.6f} rad")
    print(f"Intrinsic final mean error: {np.mean(err_intrinsic_all[:, -1]):.6f} rad")
    print(f"Additive final mean error:  {np.mean(err_additive_all[:, -1]):.6f} rad")

    # Plot mean ± std.
    plt.figure(figsize=(8, 4))
    plt.plot(time, mean_intr, label="Intrinsic SCP filter")
    plt.fill_between(time, mean_intr - std_intr, mean_intr + std_intr, alpha=0.2)

    plt.plot(time, mean_add, label="Additive EKF")
    plt.fill_between(time, mean_add - std_add, mean_add + std_add, alpha=0.2)

    plt.xlabel("time [s]")
    plt.ylabel("attitude error angle [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Boxplot of per-trial trajectory RMSE.
    plt.figure(figsize=(5, 4))
    plt.boxplot(
        [rms_intr, rms_add],
        labels=["Intrinsic\nCVXPY/SCP", "Additive\nEKF"],
    )
    plt.ylabel("trajectory RMSE [rad]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "time": time,
        "err_intrinsic_all": err_intrinsic_all,
        "err_additive_all": err_additive_all,
        "rms_intrinsic": rms_intr,
        "rms_additive": rms_add,
    }


if __name__ == "__main__":
    results = run_monte_carlo_comparison(num_trials=10)