"""Numerical integration helpers."""

from __future__ import annotations

import numpy as np

from dynamics import normalize_quat


def RK4(qt, omegat, dynamic, dt) -> np.ndarray:
    """One RK4 step for quaternion dynamics, followed by normalization."""
    qt = np.asarray(qt, dtype=float).reshape(4)
    omegat = np.asarray(omegat, dtype=float).reshape(3)

    k1x = dynamic(qt, omegat)
    k2x = dynamic(qt + dt * k1x / 2.0, omegat)
    k3x = dynamic(qt + dt * k2x / 2.0, omegat)
    k4x = dynamic(qt + dt * k3x, omegat)
    qtp1 = qt + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    return normalize_quat(qtp1)
