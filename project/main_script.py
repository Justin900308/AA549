import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from utilities import dynamics as dy
from utilities import integrator as it
import plotting as plt3D
## def simulation constants

dt = 0.02
T = 1000
nx = 4
nu = 3
nv = 4
nw = 3
## generate random noise
mu = 0
sigma = 0.5
v_traj = np.random.normal(mu, sigma, (T-1, nu))
w_traj = np.random.normal(mu, sigma, (T - 1, nw))


def traj_simulation():
    q_traj = np.zeros((T, nx))
    q_traj[:, 0] = np.ones(T)
    omega_traj = np.zeros((T-1, nu)) + v_traj * 0
    omega_traj[:,1] = np.ones(T-1) *0.05

    for t in range(T-1):
        q_traj[t + 1] = it.RK4(q_traj[t], omega_traj[t], dy.quat_dynamics, dt)

    return q_traj, omega_traj



def Estimator_sim():
    

    return








q_traj, omega_traj = traj_simulation()
plt3D.plotting3d(q_traj, omega_traj,T)