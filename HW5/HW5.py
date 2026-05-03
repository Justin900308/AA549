import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

#### constants
n = 2
m = 1
dt = 0.1
T = 100
t_list = np.linspace(0, T - 1, T) * dt
A = np.array([[1, dt],
              [0, 1]])
B = np.array([dt ** 2 / 2, dt])
mu, sigma = 0, 1
Q = np.outer(B, B)
u = np.sin(t_list * dt)  ## true control
## random variables
X0 = np.random.normal(mu, sigma, n)
v = np.random.normal(mu, sigma, T - 1)
W = np.random.normal(mu, sigma, T)


def true_dynamics(Xt, ut):
    Xtp1 = A @ Xt + B * ut
    return Xtp1


def prediction(Xt_t, ut_t, Pt_t):
    ## ut_t is the observed control
    Xtp1_t = A @ Xt_t + B * ut_t
    Ptp1_t = A @ Pt_t @ A.T + Q
    return Xtp1_t, Ptp1_t


def propagation(X0, u, v, W):
    X_list = []
    P_list = []
    X_true = np.zeros((T, n))
    X_pred_only = X_true.copy()
    X_true[0] = X0
    P_pred_only = np.zeros((T, n, n))
    P_pred_only[0] = np.eye(2) ## initial cov is given by X0

    for t in range(T - 1):
        ut = u[t]
        ut_t = ut + v[t]
        ## for true dynamics
        X_true[t + 1] = true_dynamics(X_true[t], ut)
        ## pred only
        X_pred_only[t + 1], P_pred_only[t + 1] = prediction(X_pred_only[t], ut_t, P_pred_only[t])

    X_list.append(X_true)
    X_list.append(X_pred_only)
    P_list.append(P_pred_only)
    X_error_list = []
    X_error_pred = X_true - X_pred_only
    X_error_list.append(X_error_pred)
    return X_list, X_error_list, P_list


X_list, X_error_list, P_list = propagation(X0, u, v, W)

####### plot
fig, ax = plt.subplots(figsize=(7, 5))
#### for the cov
## cov bounds (time, P)
bound_1 = 2 * np.sqrt(P_list[0][:, 0, 0])
ax.plot(t_list, X_list[1][:, 0] + bound_1, "r-.", label=rf"Cov bounds")
ax.plot(t_list, X_list[1][:, 0] - bound_1, "r-.")

#### for the traj
ax.plot(t_list, X_list[0][:, 0], "b", label=rf"True traj")
ax.plot(t_list, X_list[1][:, 0], "r-", label=rf"Pred only traj")
ax.legend()
plt.tight_layout()
plt.show()
