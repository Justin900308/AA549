import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# constants
N = 1000
P = 0.1
R = 0.4
mean = 0.0

rng = np.random.default_rng(123321)
rng = np.random.default_rng(345543)

def F_Monte_Carlo(X_1, y):
    Num = np.sum(X_1 * np.exp(-(y - X_1) ** 2 / 2 / R))
    Den = np.sum(np.exp(-(y - X_1) ** 2 / 2 / R))
    X_est_y = Num / Den
    return X_est_y  ## E[X|Y=y]


def gen_U(N):
    ran = rng.random(N)
    U = np.where(ran <= 0.5, 1.0, -1.0)
    return U


def gen_samples(N, P, R):
    X0 = rng.normal(mean, np.sqrt(P), N)
    U = gen_U(N)
    W = rng.normal(mean, np.sqrt(R), N)

    X1 = X0 + U
    Y = X1 + W
    return X1, Y


def F_lin(Y, P, R):
    return (P + 1) / (P + 1 + R) * Y


def compute_phi(Y, method):
    if method == "cubic":
        phi = np.vstack((Y, Y ** 3))  # 2 x N
    elif method == "sgn":
        sgn_Y = (Y > 0).astype(float)  # 1 if Y > 0, else 0
        phi = np.vstack((Y, sgn_Y))  # 2 x N
    return phi


def compute_E_and_Cov(X1, phi):
    N = len(X1)

    Exp_X1 = np.mean(X1)
    Exp_phi = np.reshape(np.mean(phi, axis=1), (2, 1))  # 2 x 1

    X1_centered = X1 - Exp_X1  # N
    phi_centered = phi - Exp_phi  # 2 x N

    Cov_phi = phi_centered @ phi_centered.T / N  # 2 x m
    Cov_X_phi = X1_centered @ phi_centered.T / N  # 2

    return Exp_X1, Exp_phi, Cov_phi, Cov_X_phi


def F_nonlinear(Y, X1, method):
    phi = compute_phi(Y, method)
    Exp_X1, Exp_phi, Cov_phi, Cov_X_phi = compute_E_and_Cov(X1, phi)
    X_est = Exp_X1 + Cov_X_phi @ la.inv(Cov_phi) @ (phi - Exp_phi)
    return X_est


def compute_mse(X1, X_est):
    return np.mean((X1 - X_est) ** 2)


# generate data
X1, Y = gen_samples(N, P, R)

## HW4 estimators
X_lin = F_lin(Y, P, R)
X_cubic = F_nonlinear(Y, X1, "cubic")
X_sgn = F_nonlinear(Y, X1, "sgn")
## HW7 estimation
Num_samples = [100, 1000, 10000]
X_Monte = np.zeros([3, N])
for i in range(3):
    Num_sample = Num_samples[i]
    X1_i = gen_samples(Num_sample, P, R)
    for j in range(N):
        y = Y[j]
        X_Monte[i, j] = F_Monte_Carlo(X1_i, y)


# MSEs
mse_lin = compute_mse(X1, X_lin)
mse_cubic = compute_mse(X1, X_cubic)
mse_sgn = compute_mse(X1, X_sgn)
mse_Monte = np.zeros(3)
for i in range(3):
    Num_sample = Num_samples[i]
    mse_Monte[i] = compute_mse(X1, X_Monte[i])
    print(f"MSE Monte Carlo with N1 = {Num_sample}: {mse_Monte[i]:.6f}")
print(f"MSE linear: {mse_lin:.6f}")
print(f"MSE cubic : {mse_cubic:.6f}")
print(f"MSE sgn   : {mse_sgn:.6f}")


# scatter plot
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(3):
    Num_sample = Num_samples[i]
    ax.scatter(X_Monte[i], Y, s=4, alpha=0.4, label=rf"Monte Carlo N={Num_sample}")
ax.scatter(X1, Y, s=4, alpha=0.4, label=f"Samples: (X(1),Y)")
# ax.scatter(X_lin, Y, s=4, alpha=0.4, label=f"Linear")
# ax.scatter(X_cubic, Y, s=4, alpha=0.4, label=f"Cubic")
ax.scatter(X_sgn, Y, s=4, alpha=0.4, label=f"Sign")

ax.set_xlabel(f"X(1) and f(Y)")
ax.set_ylabel(f"$Y$")
ax.set_title(f"Estimator comparison, $P={P}$, $R={R}$")
ax.legend()
plt.tight_layout()
plt.savefig("HW4.pdf")
plt.show()
