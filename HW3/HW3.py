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
    axes[i].hist(Yi, bins=interval, edgecolor='black')
    axes[i].grid(True)
    axes[i].set_ylabel('Counts')
axes[-1].set_xlabel('Yn')

# plt.tight_layout()
# plt.show()

Exp, var = Exp_var_numerical(Y, N)
print("Analytical Exp: ",Exp_analytical,"Numerical Exp: ", Exp)
print("Analytical var: ",var_analytical,"Numerical var: ", var)