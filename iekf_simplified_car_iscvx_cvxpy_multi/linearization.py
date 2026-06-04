"""Replicate the simplified-car EKF/LIEKF/ISCVX example from Barrau-Bonnabel TAC 2017.

Run:
    python main_script.py

The setup matches Sec. IV-D of the paper:
  * unicycle/simplified-car dynamics,
  * 10 m diameter circle,
  * 40 s simulation,
  * odometer/differential odometry at 100 Hz,
  * GPS position measurement at 1 Hz,
  * N = I_2,
  * Q = diag((pi/180)^2, 1e-4, 1e-4),
  * two initial heading errors: 1 deg and 45 deg,
  * initial position known.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dynamics import unicycle_dynamics_jax
from integrator import rk4_jax
import constants as ct

T = ct.T_traj_gen
dt = ct.dt_traj_gen


def linearization_fun(integrator, x_t, u_t) -> tuple:
    f = lambda x, u: integrator(unicycle_dynamics_jax, x, u,dt)
    # Compute the Jacobian of f(x, u) with respect to x (A matrix)
    A_k = jax.jacobian(lambda x: f(x, u_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    B_k = jax.jacobian(lambda u: f(x_t, u))(u_t)
    return A_k, B_k


linearization_jit = jax.jit(linearization_fun, static_argnums=0)


def linearize(x_traj, u_traj):
    ## for the dynamics
    [A_list, B_list] = jax.vmap(
        lambda x, u: linearization_jit(rk4_jax, x, u),
        in_axes=(0, 0)
    )(x_traj[0:T - 1, :], u_traj)

    Jacobians = []
    Jacobians.append(A_list)
    Jacobians.append(B_list)

    return Jacobians
