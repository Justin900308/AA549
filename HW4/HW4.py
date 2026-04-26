import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv
import scipy as sy
from matplotlib.ticker import FormatStrFormatter
import numpy.random as rnd
import scipy.linalg as la

## define global constants
N = 1000
P = 0.1
R = 0.4
mean = 0.0
# X0 ~ N(0,P), W ~ N(0,R)
X0 = rnd.normal(mean, P, N)
W = rnd.normal(mean, R, N)


def gen_U(N):
    ran = rnd.rand(N)
    U = (ran <= 1 / 2).astype(int)
    return U


def gen_Y_X1(X0, U, W):
    X1 = U + X0
    Y = X1 + W
    return Y, X1


def F_lin(Y, X1, P, R, methods):
    X_est = (P + 1) / (P + R + 1) * Y
    return X_est


def compute_E_and_Cov(X1, phi):
    N = len(X1)
    Exp_phi = np.sum(phi) / N  ## this is a vector of length 2
    Exp_X1 = np.sum(X1) / N  ## this is a scalar
    Cov_phi = np.sum((phi - Exp_phi) @ (phi - Exp_phi).T) / N  ## this is N x N matrix
    Cov_X_phi = np.sum((X1 - Exp_X1) @ (phi - Exp_phi).T) / N  ## this is N dim vector
    return Exp_X1, Exp_phi, Cov_phi, Cov_X_phi


def compute_phi(Y, methods):
    if methods == "case 1":
        first = Y
        second = np.multiply(np.multiply(Y, Y), Y)
    else:
        first = Y
        second = np.sign(Y)
    phi = np.vstack((first, second))
    return phi


def F_nolinear(Y, X1, P, R, methods):
    phi = compute_phi(Y, methods)
    Exp_X1, Exp_phi, Cov_phi, Cov_X_phi = compute_E_and_Cov(X1, phi)
    X_est = Exp_X1 + Cov_X_phi @ la.inv(Cov_phi) @ (phi - Exp_phi)
    return X_est


def scatter_plot(N, P, R, estimator, methods):
    U = gen_U(N)
    #### plot  the scatter plots
    Y, X1 = gen_Y_X1(X0, U, W)
    fig, ax = plt.subplots()
    if estimator is not None:
        X_est = estimator(Y, X1, P, R, methods)
        ax.scatter(X_est, Y, label=f"{methods}, P = {P}, R = {R} ")
        ax.legend()
        ax.set(xlabel='X est', ylabel='Y')
    else:
        ax.scatter(X1, Y, label=f"P = {P}, R = {R} ")
        ax.legend()
        ax.set(xlabel='X1', ylabel='Y')
    plt.tight_layout()
    plt.show()
    return


# ## no estimation
# scatter_plot(N, P, R, None, None)
# ## linear estimation
# scatter_plot(N, P, R, F_lin, "Linear")
## nonlinear estimation 1
scatter_plot(N, P, R, F_nolinear, "case 1")
## nonlinear estimation 2
scatter_plot(N, P, R, F_nolinear, "case 2")
