import numpy as np
import jax.numpy as jnp


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot2(theta: float) -> np.ndarray:
    """Planar rotation R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)
