"""Small integration helpers for continuous-discrete filters."""

from __future__ import annotations

import numpy as np


def covariance_euler(P: np.ndarray, A: np.ndarray, Q: np.ndarray, dt: float) -> np.ndarray:
    """Euler step for Pdot = A P + P A^T + Q."""
    Pdot = A @ P + P @ A.T + Q
    P = P + dt * Pdot
    P = 0.5 * (P + P.T)
    return P
