import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

#### constants
nx = 3
nu = 2
ny = 2
alpha = 0.03
beta = 0.5
Q = alpha ** 2 * np.eye(nx)
R = beta ** 2 * np.eye(ny)
T = 400
u = np.array([0.1, 0.02])
## generate random numbers
mu = 0
sigma = 1
v = np.random.normal(mu, sigma, (T, nx))
w = np.random.normal(mu, sigma, (T - 1, ny))
mu_ini = 0.4
sigma_ini = 1.2
# np.random.seed(12345678)
X0 = np.random.normal(mu_ini, sigma_ini, nx)


def system_output(Xt, ut, vt, wt):
    Xtp1 = np.array([Xt[0] + ut[0] * np.cos(Xt[2]) + alpha * vt[0],
                     Xt[1] + ut[0] * np.sin(Xt[2]) + alpha * vt[1],
                     Xt[2] + ut[1] + alpha * vt[2]])
    Yt = np.array([np.sqrt((Xt[0] - 1) ** 2 + Xt[1] ** 2) + beta * wt[0],
                   np.sqrt((Xt[0] + 1) ** 2 + Xt[1] ** 2) + beta * wt[1]])
    return Xtp1, Yt


def Jacobian(Xt, ut):
    At = np.array([[1, 0, -ut[0] * np.sin(Xt[2])],
                   [0, 1, ut[0] * np.cos(Xt[2])],
                   [0, 0, 1]])

    Ht = np.array(
        [[(Xt[0] - 1) / np.sqrt((Xt[0] - 1) ** 2 + Xt[1] ** 2),
          Xt[1] / np.sqrt((Xt[0] - 1) ** 2 + Xt[1] ** 2),
          0],
         [(Xt[0] + 1) / np.sqrt((Xt[0] + 1) ** 2 + Xt[1] ** 2),
          Xt[1] / np.sqrt((Xt[0] + 1) ** 2 + Xt[1] ** 2),
          0]])
    return At, Ht


def EKF_update(Xt_t, ut, Pt_t):
    At, _ = Jacobian(Xt_t, ut)
    Xtp1_t, _ = system_output(Xt_t, ut, np.zeros(nx), np.zeros(ny))
    Ptp1_t = At @ Pt_t @ At.T + Q
    return Xtp1_t, Ptp1_t


def EKF_conditioning(Xtp1_t, Ytp1, ut, Ptp1_t):
    _, Htp1 = Jacobian(Xtp1_t, ut)
    _, Y_pred = system_output(Xtp1_t, ut, np.zeros(nx), np.zeros(ny))
    ## find S
    S = Htp1 @ Ptp1_t @ Htp1.T + R
    ## find K
    K = Ptp1_t @ Htp1.T @ np.linalg.inv(S)
    ## innovation
    rtp1 = Ytp1 - Y_pred

    ## conditioning
    Xtp1_tp1 = Xtp1_t + K @ rtp1
    Ptp1_tp1 = Ptp1_t - Ptp1_t @ Htp1.T @ np.linalg.inv(S) @ Htp1 @ Ptp1_t

    return Xtp1_tp1, Ptp1_tp1, S, rtp1


def simulation(X0):
    X_traj = np.zeros((T, nx))
    X_traj[0] = X0
    X_EKF = np.zeros((T, nx))
    X_EKF[0] = np.ones(nx) * mu_ini
    Y_traj = np.zeros((T - 1, ny))
    P_EKF = np.zeros((T, nx, nx))
    P_EKF[0] = np.eye(nx) * sigma_ini ** 2
    S_traj = np.zeros((T - 1, ny, ny))
    r_traj = np.zeros((T - 1, ny))
    for t in range(T - 1):
        ## true system outputs
        X_traj[t + 1], _ = system_output(X_traj[t], u, v[t], w[t])
        ## generate the measurement since source isn't available
        _, Ytp1 = system_output(X_traj[t + 1], u, v[t], w[t])
        Y_traj[t] = Ytp1
        ## EKF
        Xtp1_t, Ptp1_t = EKF_update(X_EKF[t], u, P_EKF[t])  ## prediction
        X_EKF[t + 1], P_EKF[t + 1], S_traj[t], r_traj[t] = EKF_conditioning(Xtp1_t, Ytp1, u, Ptp1_t)  ## conditioning
    return X_traj, X_EKF, P_EKF, Y_traj, S_traj, r_traj


X_traj, X_EKF, P_EKF, Y_traj, S_traj, r_traj = simulation(X0)
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
## for xy plane
ax[0].plot(X_traj[:, 0], X_traj[:, 1], "r-", label="True traj")
ax[0].plot(X_EKF[:, 0], X_EKF[:, 1], "b-", label="EKF est")
ax[0].plot(X_traj[0, 0], X_traj[0, 1], "ro", label="Start", markersize=10)
ax[0].plot(X_traj[-1, 0], X_traj[-1, 1], "go", label="End", markersize=10)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
## for heading
t_list = np.linspace(0, T - 1, T)
ax[1].plot(t_list, X_traj[:, 2] * 180 / np.pi, "r-", label="True traj")
ax[1].plot(t_list, X_EKF[:, 2] * 180 / np.pi, "b-", label="EKF est")
ax[1].set_xlabel("t")
ax[1].set_ylabel("heading")
for i in range(2):
    ax[i].legend()
    ax[i].grid()
plt.tight_layout()
plt.show()

#### check the condition
f = np.zeros(2)
for t in range(T - 1):
    St = S_traj[t]
    rt = r_traj[t]
    for j in range(ny):
        f[j] += rt[j] / np.sqrt(St[j, j])
f = f / T
print(rf"{np.abs(f[0])}, {np.abs(f[1])}, bounds = {2 / np.sqrt(T - 1)}")
