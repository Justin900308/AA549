import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

#### constants
n = 2
m = 1
n_aug = 3
dt = 0.1
T = 100
t_list = np.linspace(0, T - 1, T) * dt
A = np.array([[1, dt],
              [0, 1]])
A_aug = np.array([[1, dt, -dt ** 2 / 2],
                  [0, 1, -dt],
                  [0, 0, 1]])
B = np.array([dt ** 2 / 2, dt])
B_aug = np.array([dt ** 2 / 2, dt, 0])
C = np.array([[1, 0]])
C_aug = np.array([[1, 0, 0]])
mu, sigma = 0, 1
Q = np.outer(B, B)
Q_aug = np.outer(B_aug, B_aug)
R = 1
u = np.sin(t_list)  ## true control
## random variables
X0 = np.random.normal(mu, sigma, n)
v = np.random.normal(mu, sigma, T - 1)
W = np.random.normal(mu, sigma, T)
b = np.random.normal(mu, sigma, 1)

## check if observable
Obs_C = np.vstack((C_aug, np.vstack((C_aug @ A_aug, C_aug @ A_aug @ A_aug))))
rank = np.linalg.matrix_rank(Obs_C)
print(rf"rank {rank} larger or equal to 3, observable")


def true_dynamics(Xt, ut):
    Xtp1 = A @ Xt + B * ut
    return Xtp1


def prediction(A, B, Q, Xt_t, ut_t, Pt_t):
    ## ut_t is the observed control
    Xtp1_t = A @ Xt_t + B * ut_t
    Ptp1_t = A @ Pt_t @ A.T + Q
    return Xtp1_t, Ptp1_t


def condition(C, Xtp1_t, Ytp1, Ptp1_t):
    ## compute S
    S = C @ Ptp1_t @ C.T + R
    ## compute K
    K = Ptp1_t @ C.T / S
    ## innovation
    innov = Ytp1 - C @ Xtp1_t

    ## conditioning
    Xtp1_tp1 = Xtp1_t + (K * innov).reshape(-1)
    Ptp1_tp1 = Ptp1_t - Ptp1_t @ C.T @ C @ Ptp1_t / S

    return Xtp1_tp1, Ptp1_tp1


def propagation(X0, u, v, W):
    X_list = []
    P_list = []
    X_true = np.zeros((T, n))
    X_pred_only = X_true.copy()
    X_KF = X_true.copy()
    X_KF_aug = np.zeros((T, n_aug))
    X_true[0] = X0
    P_pred_only = np.zeros((T, n, n))
    P_pred_only[0] = np.eye(2)  ## initial cov is given by X0
    P_KF = P_pred_only.copy()
    P_KF_aug = np.zeros((T, n_aug, n_aug))
    P_KF_aug[0] = np.eye(n_aug)
    for t in range(T - 1):
        ut = u[t]
        ut_t = ut + v[t]
        ## for true dynamics
        X_true[t + 1] = true_dynamics(X_true[t], ut)
        #### pred only
        X_pred_only[t + 1], P_pred_only[t + 1] = prediction(A, B, Q, X_pred_only[t], ut_t, P_pred_only[t])

        #### For KF
        X_pred_KF, P_pred_KF = prediction(A, B, Q, X_KF[t], ut_t, P_KF[t])  ## prediction
        Ytp1 = C @ X_true[t + 1] + W[t + 1]
        X_KF[t + 1], P_KF[t + 1] = condition(C, X_pred_KF, Ytp1, P_pred_KF)  ## conditioning

        #### For aug KF
        ut_t_b = ut_t + b
        X_pred_KF_aug, P_pred_KF_aug = prediction(A_aug, B_aug, Q_aug, X_KF_aug[t], ut_t_b, P_KF_aug[t])  ## prediction
        Ytp1_aug = C_aug @ np.hstack((X_true[t + 1], b)) + W[t + 1]
        X_KF_aug[t + 1], P_KF_aug[t + 1] = condition(C_aug, X_pred_KF_aug, Ytp1_aug, P_pred_KF_aug)  ## conditioning

    X_list.append(X_true)
    X_list.append(X_pred_only)
    X_list.append(X_KF)
    X_list.append(X_KF_aug)
    P_list.append(P_pred_only)
    P_list.append(P_KF)
    X_error_list = []
    X_error_pred = X_true - X_pred_only
    X_error_KF = X_true - X_KF
    X_error_list.append(X_error_pred)
    X_error_list.append(X_error_KF)
    return X_list, X_error_list, P_list


X_list, X_error_list, P_list = propagation(X0, u, v, W)

####### plot
fig, ax = plt.subplots(2, 1, figsize=(7, 5))
#### for the cov
## cov bounds (time, P)
bound_pred = 2 * np.sqrt(P_list[0][:, 0, 0])
ax[0].plot(t_list, X_list[1][:, 0] + bound_pred, "r-.", label=rf"Cov pred bounds")
ax[0].plot(t_list, X_list[1][:, 0] - bound_pred, "r-.")
bound_KF = 2 * np.sqrt(P_list[1][:, 0, 0])
ax[1].plot(t_list, X_list[2][:, 0] + bound_KF, "r-.", label=rf"Cov KF bounds")
ax[1].plot(t_list, X_list[2][:, 0] - bound_KF, "r-.")

#### for the traj
for i in range(2):
    ax[i].plot(t_list, X_list[0][:, 0], "b", label=rf"True traj")
    ax[i].plot(t_list, X_list[i + 1][:, 0], "r-", label=rf"Est traj")
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xlabel(r"$t$")
    ax[i].set_ylabel(r"$X_1,\; \hat{X}_1$")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(7, 5))
b_list = np.ones(T) * b
ax[0].plot(t_list, X_list[0][:, 0], "b", label=rf"True traj")
ax[0].plot(t_list, X_list[3][:, 0], "r-", label=rf"Est traj")
ax[1].plot(t_list, b_list, "b", label=rf"True bias")
ax[1].plot(t_list, X_list[3][:, 2], "r-", label=rf"Est bias")
for i in range(2):
    ax[i].grid(True)
    ax[i].set_xlabel(r"$t$")
    ax[i].legend()
ax[0].set_ylabel(r"$X_1,\; \hat{X}_1$")
ax[1].set_ylabel(r"$b,\; \hat{b}$")
plt.tight_layout()
plt.show()
