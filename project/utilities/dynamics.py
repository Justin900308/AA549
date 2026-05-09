import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import jax.numpy as jnp

def quat_conj(q: jnp.array) -> jnp.array:
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(p: jnp.array, q: jnp.array) -> jnp.array:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q: jnp.array, v: jnp.array) -> jnp.array:
    # rotate 3-vector v from body to inertial using quaternion q
    v_quat = jnp.concatenate([jnp.array([0.0]), v])
    v_rot = quat_mul(quat_mul(q, v_quat), quat_conj(q))
    return v_rot[1:]

def quat_dynamics(qt: jnp.ndarray, omegat: jnp.ndarray) -> jnp.ndarray:
    qw, qx, qy, qz = qt
    wx, wy, wz = omegat

    Omega = jnp.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0, wx],
        [wz,  wy, -wx, 0.0],
    ])

    qdot = 0.5 * Omega @ qt
    return qdot



