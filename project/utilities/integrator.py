import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from project.utilities import dynamics


## integrator for the dynamics and the gyro measure
def RK4(qt, omegat, dynamic, dt) -> jnp.ndarray:
    k1x = dynamic(qt, omegat)
    k2x = dynamic(qt + dt * k1x / 2, omegat)
    k3x = dynamic(qt + dt * k2x / 2, omegat)
    k4x = dynamic(qt + dt * k3x, omegat)
    qtp1 = qt + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6


    qtp1 = qtp1 / jnp.linalg.norm(qtp1, 2)

    return qtp1
