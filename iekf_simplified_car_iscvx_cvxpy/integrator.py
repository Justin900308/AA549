"""Small integration helpers for continuous-discrete filters."""

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp
import dynamics as dy

def covariance_euler(P: np.ndarray, A: np.ndarray, Q: np.ndarray, dt: float) -> np.ndarray:
    """Euler step for Pdot = A P + P A^T + Q."""
    Pdot = A @ P + P @ A.T + Q
    P = P + dt * Pdot
    P = 0.5 * (P + P.T)
    return P


def rk4(dynamic,z: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step for the unicycle state."""
    z = np.asarray(z, dtype=float).reshape(3)
    u = np.asarray(u, dtype=float).reshape(2)
    k1 = dynamic(z, u)
    k2 = dynamic(z + 0.5 * dt * k1, u)
    k3 = dynamic(z + 0.5 * dt * k2, u)
    k4 = dynamic(z + dt * k3, u)
    zp1 = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    zp1[0] = dy.wrap_angle(zp1[0])
    return zp1


def rk4_jax(dynamic,z, u, dt) -> jnp.ndarray:
    """One RK4 step for the unicycle state."""
    z = jnp.asarray(z, dtype=float).reshape(3)
    u = jnp.asarray(u, dtype=float).reshape(2)
    k1 = dynamic(z, u)
    k2 = dynamic(z + 0.5 * dt * k1, u)
    k3 = dynamic(z + 0.5 * dt * k2, u)
    k4 = dynamic(z + dt * k3, u)
    zp1 = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    zp1 = zp1.at[0].set(dy.wrap_angle_jax(zp1[0]))

    return zp1


RK_jit = jax.jit(rk4_jax, static_argnums=(0,))