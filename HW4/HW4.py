import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv
import scipy as sy
from matplotlib.ticker import FormatStrFormatter
import numpy.random as rnd

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


def F_lin(Y, P, R):
    X_est = (P + 1) / (P + R + 1) * Y
    return X_est


def scatter_plot(N, P, R, estimator, est_name):
    U = gen_U(N)
    #### plot  the scatter plots
    Y, X1 = gen_Y_X1(X0, U, W)
    fig, ax = plt.subplots()
    if estimator is not None:
        X_est = estimator(Y, P, R)
        ax.scatter(X_est, Y, label=f"{est_name}, P = {P}, R = {R} ")
        ax.legend()
        ax.set(xlabel='X est', ylabel='Y')
    else:
        ax.scatter(X1, Y, label=f"P = {P}, R = {R} ")
        ax.legend()
        ax.set(xlabel='X1', ylabel='Y')
    plt.tight_layout()
    plt.show()
    return


## no estimation
scatter_plot(N, P, R, None, None)
## linear estimation
scatter_plot(N, P, R, F_lin, "Linear")
