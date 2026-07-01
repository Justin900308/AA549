"""Flat-earth inertial navigation dynamics and SE_2(3) utilities.

State convention
----------------
A state is represented by

    R : body -> inertial rotation matrix
    v : inertial velocity
    p : inertial position

and embedded in the double homogeneous matrix Lie group SE_2(3):

    chi = [[R, v, p],
           [0, 1, 0],
           [0, 0, 1]].

The physical model follows Barrau--Bonnabel, TAC 2017, Sec. V:

    R_dot = R (omega)_x,
    v_dot = g + R u,
    p_dot = v,

with landmark measurements y_i = R^T (p_i - p).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


Array = np.ndarray


@dataclass
class State:
    R: Array
    v: Array
    p: Array

    def copy(self) -> "State":
        return State(self.R.copy(), self.v.copy(), self.p.copy())


@dataclass
class IMUInput:
    gyro: Array
    acc: Array


GRAVITY = np.array([0.0, 0.0, -9.82])
LANDMARKS = np.array(
    [
        [0.0, 2.0, 2.0],
        [-2.0, -2.0, -2.0],
        [2.0, -2.0, -2.0],
    ],
    dtype=float,
)


def skew(a: Array) -> Array:
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        dtype=float,
    )


def vee(A: Array) -> Array:
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def so3_exp(phi: Array) -> Array:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-12:
        return np.eye(3) + K + 0.5 * K @ K
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / theta**2
    return np.eye(3) + A * K + B * K @ K


def so3_log(R: Array) -> Array:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return 0.5 * vee(R - R.T)
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)


def so3_left_jacobian(phi: Array) -> Array:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-10:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * K @ K
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta**2 * K
        + (theta - np.sin(theta)) / theta**3 * K @ K
    )


def so3_left_jacobian_inv(phi: Array) -> Array:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-10:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * K @ K
    A = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * K + A * K @ K


def se23_wedge(xi: Array) -> Array:
    xi = np.asarray(xi, dtype=float).reshape(9)
    phi, nu, rho = xi[:3], xi[3:6], xi[6:9]
    X = np.zeros((5, 5), dtype=float)
    X[:3, :3] = skew(phi)
    X[:3, 3] = nu
    X[:3, 4] = rho
    return X


def se23_exp(xi: Array) -> Array:
    xi = np.asarray(xi, dtype=float).reshape(9)
    phi, nu, rho = xi[:3], xi[3:6], xi[6:9]
    R = so3_exp(phi)
    J = so3_left_jacobian(phi)
    chi = np.eye(5)
    chi[:3, :3] = R
    chi[:3, 3] = J @ nu
    chi[:3, 4] = J @ rho
    return chi


def se23_log(chi: Array) -> Array:
    chi = np.asarray(chi, dtype=float).reshape(5, 5)
    phi = so3_log(chi[:3, :3])
    Jinv = so3_left_jacobian_inv(phi)
    nu = Jinv @ chi[:3, 3]
    rho = Jinv @ chi[:3, 4]
    return np.r_[phi, nu, rho]


def state_to_chi(state: State) -> Array:
    chi = np.eye(5)
    chi[:3, :3] = state.R
    chi[:3, 3] = state.v
    chi[:3, 4] = state.p
    return chi


def chi_to_state(chi: Array) -> State:
    return State(R=chi[:3, :3].copy(), v=chi[:3, 3].copy(), p=chi[:3, 4].copy())


def se23_inv(chi: Array) -> Array:
    R = chi[:3, :3]
    v = chi[:3, 3]
    p = chi[:3, 4]
    out = np.eye(5)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ v
    out[:3, 4] = -R.T @ p
    return out


def standard_retract(state: State, xi: Array) -> State:
    """MEKF retraction on SO(3) x R^6: (Exp(phi)R, v+dv, p+dp)."""
    xi = np.asarray(xi, dtype=float).reshape(9)
    return State(
        R=so3_exp(xi[:3]) @ state.R,
        v=state.v + xi[3:6],
        p=state.p + xi[6:9],
    )


def right_retract(state: State, xi: Array) -> State:
    """Right-invariant SE_2(3) retraction used by the paper's IEKF: Exp(xi) chi."""
    return chi_to_state(se23_exp(xi) @ state_to_chi(state))


def right_inv_retract(base: State, state: State) -> Array:
    """Return xi such that state = Exp(xi) base, i.e. Log(chi_state chi_base^{-1})."""
    return se23_log(state_to_chi(state) @ se23_inv(state_to_chi(base)))


def propagate_state(state: State, omega: IMUInput, dt: float, g: Array = GRAVITY) -> State:
    """Discrete integration used in the UKFM benchmark/source.

    R_{k+1} = R_k Exp(gyro*dt)
    v_{k+1} = v_k + (R_k acc + g) dt
    p_{k+1} = p_k + v_k dt + 1/2 (R_k acc + g) dt^2
    """
    acc_world = state.R @ np.asarray(omega.acc).reshape(3) + g
    return State(
        R=state.R @ so3_exp(np.asarray(omega.gyro).reshape(3) * dt),
        v=state.v + acc_world * dt,
        p=state.p + state.v * dt + 0.5 * acc_world * dt**2,
    )


def measurement_model(state: State, landmarks: Array = LANDMARKS) -> Array:
    """Stack y_i = R^T (p_i - p) for all landmarks."""
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 3)
    return np.concatenate([state.R.T @ (lm - state.p) for lm in landmarks])


def iekf_residual(state_hat: State, y: Array, landmarks: Array = LANDMARKS) -> Array:
    """Paper RIEKF residual stack: Rhat y_i + phat - p_i.

    This is zero at the correct state and is the top block of chi_hat Y_i - d_i.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 3)
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 3)
    return np.concatenate([state_hat.R @ yi + state_hat.p - lm for yi, lm in zip(y, landmarks)])


def ekf_A_matrix(state_hat: State, omega: IMUInput) -> Array:
    """Continuous MEKF F_t from the paper/source, order [att, vel, pos]."""
    A = np.zeros((9, 9), dtype=float)
    A[3:6, 0:3] = -skew(state_hat.R @ omega.acc)
    A[6:9, 3:6] = np.eye(3)
    return A


def iekf_A_matrix(g: Array = GRAVITY) -> Array:
    """Continuous IEKF A_t from the paper, order [att, vel, pos]."""
    A = np.zeros((9, 9), dtype=float)
    A[3:6, 0:3] = skew(g)
    A[6:9, 3:6] = np.eye(3)
    return A


def ekf_H_matrix(state_hat: State, landmarks: Array = LANDMARKS) -> Array:
    """MEKF landmark Jacobian matching the UKFM/source convention."""
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 3)
    H = np.zeros((3 * len(landmarks), 9), dtype=float)
    for i, lm in enumerate(landmarks):
        sl = slice(3 * i, 3 * (i + 1))
        H[sl, 0:3] = state_hat.R.T @ skew(lm - state_hat.p)
        H[sl, 6:9] = -state_hat.R.T
    return H


def iekf_H_matrix(landmarks: Array = LANDMARKS) -> Array:
    """Paper IEKF H = [(p_i)_x, 0, -I] stacked."""
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 3)
    H = np.zeros((3 * len(landmarks), 9), dtype=float)
    for i, lm in enumerate(landmarks):
        sl = slice(3 * i, 3 * (i + 1))
        H[sl, 0:3] = skew(lm)
        H[sl, 6:9] = -np.eye(3)
    return H


def iekf_Q_hat(state_hat: State, Q_base: Array) -> Array:
    """Qhat matrix from paper Sec. V-B3."""
    Q_base = np.asarray(Q_base, dtype=float).reshape(9, 9)
    M = np.zeros((9, 9), dtype=float)
    R = state_hat.R
    M[0:3, 0:3] = R
    M[3:6, 0:3] = skew(state_hat.v) @ R
    M[3:6, 3:6] = R
    M[6:9, 0:3] = skew(state_hat.p) @ R
    M[6:9, 6:9] = R
    return M @ Q_base @ M.T


def block_diag_rotation_noise(state_hat: State, obs_cov_body: Array, n_landmarks: int) -> Array:
    """Nhat = blockdiag(Rhat Cov(V_i) Rhat^T)."""
    obs_cov_body = np.asarray(obs_cov_body, dtype=float).reshape(3, 3)
    N = np.zeros((3 * n_landmarks, 3 * n_landmarks), dtype=float)
    block = state_hat.R @ obs_cov_body @ state_hat.R.T
    for i in range(n_landmarks):
        N[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = block
    return N


def numerical_jacobian(fun, dim: int, eps: float = 1e-6) -> Array:
    f0 = np.asarray(fun(np.zeros(dim)), dtype=float).reshape(-1)
    J = np.zeros((f0.size, dim), dtype=float)
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = eps
        J[:, i] = (np.asarray(fun(e), dtype=float).reshape(-1) - np.asarray(fun(-e), dtype=float).reshape(-1)) / (2.0 * eps)
    return J


def iscvx_prior_residual_jacobian(state_pred: State, state_iter: State) -> Array:
    """Jacobian of Log(chi(candidate) chi_pred^{-1}) at candidate=Exp(eta) chi_iter."""
    def f(eta):
        cand = right_retract(state_iter, eta)
        return right_inv_retract(state_pred, cand)

    return numerical_jacobian(f, 9)


def iscvx_measurement_jacobian(state_iter: State, landmarks: Array = LANDMARKS) -> Array:
    """Jacobian of h(Exp(eta) chi_iter) at eta=0."""
    def f(eta):
        cand = right_retract(state_iter, eta)
        return measurement_model(cand, landmarks)

    return numerical_jacobian(f, 9)


def attitude_error_deg(R_true: Array, R_hat: Array) -> float:
    return float(np.linalg.norm(so3_log(R_true.T @ R_hat)) * 180.0 / np.pi)


def position_error(p_true: Array, p_hat: Array) -> float:
    return float(np.linalg.norm(np.asarray(p_true) - np.asarray(p_hat)))
