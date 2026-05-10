"""Plot helpers for the simplified-car replication."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_simplified_car_cases(results_by_case, save_path: str | None = None):
    """Create a Fig. 1 style comparison: trajectory, heading error, position error."""
    n_cases = len(results_by_case)
    fig, axes = plt.subplots(3, n_cases, figsize=(6.2 * n_cases, 9.0), constrained_layout=True)
    if n_cases == 1:
        axes = axes.reshape(3, 1)

    for j, res in enumerate(results_by_case):
        t = res["time"]
        true = res["true"]
        ekf = res["ekf"]
        liekf = res["liekf"]
        init_deg = res["initial_heading_error_deg"]

        ax = axes[0, j]
        ax.plot(true[:, 1], true[:, 2], "-", label="True trajectory")
        ax.plot(ekf[:, 1], ekf[:, 2], "--", label="EKF estimate")
        ax.plot(liekf[:, 1], liekf[:, 2], "-", label="LIEKF estimate")
        ax.set_title(f"Estimated trajectory, $e_\\theta(0)={init_deg:g}^\\circ$")
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax = axes[1, j]
        ax.plot(t, res["heading_error_ekf_deg"], "--", label="EKF")
        ax.plot(t, res["heading_error_liekf_deg"], "-", label="LIEKF")
        ax.set_title("Attitude error (degrees)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("|heading error| (deg)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax = axes[2, j]
        ax.plot(t, res["position_error_ekf_m"], "--", label="EKF")
        ax.plot(t, res["position_error_liekf_m"], "-", label="LIEKF")
        ax.set_title("Position error (m)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position error (m)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig, axes
