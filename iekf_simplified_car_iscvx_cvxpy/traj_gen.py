import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
import jax
import cvxpy as cp
import constants as ct

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import linearization as lr
import dynamics as dy
import integrator as it
from matplotlib.patches import Circle
pi = np.pi

## obstacles
num_obs = ct.num_obs
obs = ct.obs
obs_r = ct.obs_r

## global variables
n = ct.n
m = ct.m
dt = ct.dt_traj_gen
T = ct.T_traj_gen
time_traj = np.linspace(0, T - 1, T)
x_0 = np.zeros(n)
x_des = ct.z_des
x_traj = np.zeros([T, n])
x_traj[0] = x_0
u_traj = np.zeros([T - 1, m])


def cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s, lambd):
    ## terminal cost
    f0 = 100 * cp.norm(x_traj[T - 1] + d[T - 1] - x_des, 2)
    f_prox = 0
    ## running cost
    for t in range(T - 1):
        ## for minimum energy control
        f0 += 1 * cp.sum_squares(u_traj[t] + w[t])
        ## for dynamics constraints
        f0 += 1000 * cp.norm(v[t], 1)
        # ## for obs
        for j in range(num_obs):
            f0 += 1000 * cp.abs(s[t, j])
    ## regularization
    f_prox += f0
    for t in range(T - 1):
        f_prox += lambd * cp.norm(w[t], 2)

    return f0, f_prox


def solve_subproblem(Jacobians, trajs, x_des, lambd, iter) -> tuple:
    ## dynamics
    A_list = Jacobians[0]
    B_list = Jacobians[1]
    x_traj = trajs[0]
    u_traj = trajs[1]
    f_traj = trajs[2]
    d = cp.Variable([T, n])
    w = cp.Variable([T - 1, m])
    v = cp.Variable([T, n])
    s = cp.Variable([T, num_obs], nonneg=True)
    ## starting constraints
    constraints = [d[0] == np.zeros(n)]
    constraints.append(x_traj[T - 1] + d[T - 1] == x_des)
    for t in range(T - 1):
        x_t = x_traj[t]
        x_tp1 = x_traj[t + 1]
        u_t = u_traj[t]
        w_t = w[t]
        d_t = d[t]
        d_tp1 = d[t + 1]
        v_t = v[t]
        s_t = s[t]
        A_t = A_list[t]
        B_t = B_list[t]
        f_t = f_traj[t]
        # f_t = Integrator.RK4(dt, x_t, u_t, W_t)
        # print(f_traj[t], f_t)
        constraints.append(x_tp1 + d_tp1 == A_t @ d_t + B_t @ w_t + f_t + 1 * np.diag(np.ones(n)) @ v_t)

        #
        ## obs constraints
        if iter>0:
            for j in range(num_obs):
                obs_j = obs[j]
                h_j = obs_r ** 2 - LA.norm(x_t[1:3] - obs_j, 2) ** 2
                a = - 2 * (x_t[1:3] - obs_j)
                LHS = h_j + a @ d_t[1:3]  ## obs constraints
                constraints.append(LHS <= s_t[j] * 0)
        #
    # f0 = cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s)
    f0, f_prox = cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s, lambd)
    problem = cp.Problem(cp.Minimize(f_prox), constraints)
    problem.solve(solver=cp.CLARABEL)
    d_traj = d.value
    w_traj = w.value
    v_val = v.value
    true_cost = f0.value
    subproblem_cost = problem.value

    return d_traj, w_traj, v_val, true_cost, subproblem_cost


def penalty_regulation(true_cost_list, iter, lambd):
    if true_cost_list[iter] > true_cost_list[iter - 1]:
        lambd *= 1.5
    return lambd


def traj_gen(x_traj, u_traj) -> tuple:
    max_iter = 15
    lambd = 0.2
    true_cost_old = 0
    true_cost_list = np.zeros(max_iter)
    for iter in range(max_iter):

        ## get current nonlinear states
        f_traj = jax.vmap(
            lambda x, u: it.RK_jit(dy.unicycle_dynamics_jax, x, u, dt),
            in_axes=(0, 0)
        )(x_traj[0:T - 1, :], u_traj)

        ## get the Jacobians
        Jacobians = lr.linearize(x_traj, u_traj)
        trajs = [x_traj, u_traj, f_traj]
        [d_traj, w_traj, v_traj, true_cost, subproblem_cost] = solve_subproblem(Jacobians, trajs, x_des, lambd,
                                                                                iter)
        print("Traj upt iteration:  ", iter + 1, "true cost", true_cost, "subproblem cost:    ", subproblem_cost,
              "lambda", lambd)
        true_cost_list[iter] = true_cost

        ## update the penalty parameters
        if iter != 0:
            lambd = penalty_regulation(true_cost_list, iter, lambd)

        v_norm_traj = np.zeros(T)
        for t in range(T):
            v_norm_traj[t] = LA.norm(v_traj[t], 2)
        ## update
        if d_traj.any() != None:
            x_traj += d_traj
            u_traj += w_traj
        print("cost diff:   ", np.abs(true_cost_old - true_cost))
        if np.abs(true_cost_old - true_cost) <= 0.001:
            print("Sol converged")
            plt.plot(time_traj, v_norm_traj, ".")
            plt.show()
            break

        # if np.abs(subproblem_cost_old - subproblem_cost) <= 0.01 or iter > 25:
        #     plotting3d_fcn(x_traj, Q_traj)
        true_cost_old = true_cost
    ## plotting
    fig, ax = plt.subplots(figsize = (10,5))
    ax.plot(x_des[1], x_des[2], "r.", label="x_des")
    ax.plot(x_0[1], x_0[2], "g.", label="x_ini")
    for i in range(num_obs):
        circle = Circle((obs[i,0], obs[i,1]), obs_r, fill=False, linewidth=2)
        ax.add_patch(circle)
    for t in range(T-1):
        ax.plot(x_traj[t,1],x_traj[t,2],'b.')
        ax.set_xlim([0,10])
        ax.set_ylim([0,6])
        plt.pause(0.1)
    ax.plot(x_traj[:, 1], x_traj[:, 2], "b-")
    ax.legend()
    plt.show()

    return x_traj, u_traj, Jacobians


#
[x_traj, u_traj, Jacobians] = traj_gen(x_traj, u_traj)
