import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv
import scipy as sy
from matplotlib.ticker import FormatStrFormatter
import numpy.random as rnd

## constant
N = 10000
n_list = [1, 2, 10, 100]


## generate [Nxn] samples
def gen_x(N, n):
    U = rnd.rand(N, n)
    x = (U <= 2 / 3).astype(int)
    return x


## get Yi (N dim vector) from x
def gen_Yi(x):
    Yi = np.sum(x, axis=1)
    return Yi


def Exp_var_numerical(Y, N):
    Exp = np.zeros(4)
    var = np.zeros(4)
    for i in range(4):
        Yi = Y[i]
        Exp[i] = np.sum(Yi) / N
        square_part = np.multiply(Yi - Exp[i], Yi - Exp[i])
        var[i] = np.sum(square_part) / N
    return Exp, var


def gen_Z(Y,n):
    Y_last = Y[-1] / n
    exp_x = 2 / 3
    var_x = 2 / 9
    Z = (Y_last - exp_x) * np.sqrt(n / var_x)
    return Z

def normal_pdf(grid_num):
    pdf = np.zeros(grid_num)
    grid = np.linspace(-5, 5, grid_num)
    for i in range(grid_num):
        pdf[i] = (1 / np.sqrt(2 * np.pi)) * np.exp(-grid[i] ** 2 / 2)
    return pdf,grid


Y = np.zeros([4, N])
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 8))
Exp_analytical = np.zeros(4)
var_analytical = np.zeros(4)
for i in range(4):
    n = n_list[i]
    ## get analytical exp and var
    Exp_analytical[i] = n * 2 / 3
    var_analytical[i] = n * 2 / 9
    ## generate X
    x = gen_x(N, n)
    ## generate Yi (N realizations of yn)
    Yi = gen_Yi(x)
    Y[i] = Yi
    ## plotting
    interval = np.linspace(-0.5, n + 0.5, n + 2)
    weights = np.ones_like(Yi)/ len(Yi)
    axes[i].hist(Yi, bins=interval, weights=weights, edgecolor='black')
    axes[i].grid(True)
    axes[i].set_ylabel('Prob')
axes[-1].set_xlabel('Yn')
plt.tight_layout()
plt.show()
fig.savefig(f'Y')

Exp, var = Exp_var_numerical(Y, N)
print("Analytical Exp: ", Exp_analytical, "Numerical Exp: ", Exp)
print("Analytical var: ", var_analytical, "Numerical var: ", var)
Z = gen_Z(Y, n_list[-1])
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
# weights_z = np.ones_like(Z)/ len(Z)
ax.hist(Z,bins = 35,density=True, edgecolor='black',label='Z')
gaussian_pdf,grid = normal_pdf(1000)
ax.plot(grid, gaussian_pdf,label = "Gaussian")
ax.set_ylabel('Prob')
ax.set_xlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(f'Z')

