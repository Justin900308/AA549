"""Covariance integration helpers."""

from __future__ import annotations

import numpy as np


def covariance_euler(P: np.ndarray, A: np.ndarray, Q: np.ndarray, dt: float) -> np.ndarray:
    """Euler step for Pdot = A P + P A^T + Q."""
    Pdot = A @ P + P @ A.T + Q
    P = P + dt * Pdot
    P = 0.5 * (P + P.T)
    return P


def spd_project(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project a symmetric matrix to the SPD cone by eigenvalue clipping."""
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)
