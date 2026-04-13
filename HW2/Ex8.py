import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv
import scipy as sy
from matplotlib.ticker import FormatStrFormatter

t_all = 0
with open('t.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        t = float(lines[0])
        t_all = np.hstack((t_all, t))
t_all = np.array(t_all[1:])
T = len(t_all)

Y = np.zeros([T, 2])
with open('Y.csv', mode='r') as file:
    csvFile = csv.reader(file)
    t = 0
    for lines in csvFile:
        Y[t] = np.array([lines[0], lines[1]])
        t += 1


def H_fcn(ti):
    Hi = np.array([[ti ** 2, ti, 1, 0, 0, 0],
                   [0, 0, 0, ti ** 2, ti, 1]])
    return Hi


def solve_with_cvxpy(Y):
    X = cp.Variable(6)
    f = 0
    for i in range(T):
        ti = t_all[i]
        Hi = H_fcn(ti)
        f += cp.sum_squares(Y[i] - Hi @ X)

    problem = cp.Problem(cp.Minimize(f), [])
    problem.solve(solver=cp.CLARABEL)
    print(X.value)
    return X.value


def solve_analytic(Y):
    ## matrices at t = 0
    A = H_fcn(t_all[0]).T @ H_fcn(t_all[0])
    B = H_fcn(t_all[0]).T @ Y[0]
    for i in range(T - 1):
        ti = t_all[i + 1]
        Hi = H_fcn(ti)
        A += Hi.T @ Hi
        B += Hi.T @ Y[i+1]
    X_val = sy.linalg.inv(A) @ B.T
    return X_val


# X_val = solve_with_cvxpy(Y)
X_val = solve_analytic(Y)
Y_est = np.zeros([T, 2])
for i in range(T):
    ti = t_all[i]
    Hi = np.array([[ti ** 2, ti, 1, 0, 0, 0],
                   [0, 0, 0, ti ** 2, ti, 1]])
    Y_est[i] = Hi @ X_val

# Creates a figure with 2 rows and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
axes[0, 0].plot(t_all, Y[:, 0])
axes[0, 0].set_ylabel('x pos (raw)')
axes[1, 0].plot(t_all, Y[:, 1])
axes[1, 0].set_ylabel('y pos (raw)')
axes[1, 0].set_xlabel('Time')
axes[0, 1].plot(t_all, Y_est[:, 0])
axes[0, 1].set_ylabel('x pos (est)')
axes[1, 1].plot(t_all, Y_est[:, 1])
axes[1, 1].set_ylabel('y pos (est)')
axes[1, 1].set_xlabel('Time')
fig.savefig(f'raw_vs_est')
t10 = 10
H10 = np.array([[t10 ** 2, t10, 1, 0, 0, 0],
                [0, 0, 0, t10 ** 2, t10, 1]])
Y_est_10 = H10 @ X_val
print(f"pos at t = 10:{Y_est_10}")


plt.tight_layout()  # Fixes overlapping labels
plt.show()
