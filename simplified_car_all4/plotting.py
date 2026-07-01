"""Plot helpers for the simplified-car four-method comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_simplified_car_cases(results_by_case, save_path: str | None = None):
    """Create a Fig. 1 style comparison: trajectory, heading error, position error."""
    n_cases = len(results_by_case)
    fig, axes = plt.subplots(3, n_cases, figsize=(6.8 * n_cases, 9.0), constrained_layout=True)
    if n_cases == 1:
        axes = axes.reshape(3, 1)

    for j, res in enumerate(results_by_case):
        t = res["time"]
        true = res["true"]
        init_deg = res["initial_heading_error_deg"]
        landmarks = res["landmarks"]

        ax = axes[0, j]
        ax.plot(true[:, 1], true[:, 2], "-", label="True trajectory")
        ax.plot(res["ekf"][:, 1], res["ekf"][:, 2], "--", label="EKF/GPS")
        ax.plot(res["liekf"][:, 1], res["liekf"][:, 2], "-", label="LIEKF/GPS")
        ax.plot(res["iscvx"][:, 1], res["iscvx"][:, 2], ":", linewidth=2.0, label="ISCVX/GPS")
        ax.plot(res["riekf"][:, 1], res["riekf"][:, 2], "-.", label="RIEKF/landmarks")
        ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="x", s=60, label="landmarks")
        ax.set_title(f"Estimated trajectory, $e_\\theta(0)={init_deg:g}^\\circ$")
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(loc="best", fontsize=7)

        ax = axes[1, j]
        ax.plot(t, res["heading_error_ekf_deg"], "--", label="EKF")
        ax.plot(t, res["heading_error_liekf_deg"], "-", label="LIEKF")
        ax.plot(t, res["heading_error_iscvx_deg"], ":", linewidth=2.0, label="ISCVX")
        ax.plot(t, res["heading_error_riekf_deg"], "-.", label="RIEKF")
        ax.set_title("Attitude error (degrees)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("|heading error| (deg)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=7)

        ax = axes[2, j]
        ax.plot(t, res["position_error_ekf_m"], "--", label="EKF")
        ax.plot(t, res["position_error_liekf_m"], "-", label="LIEKF")
        ax.plot(t, res["position_error_iscvx_m"], ":", linewidth=2.0, label="ISCVX")
        ax.plot(t, res["position_error_riekf_m"], "-.", label="RIEKF")
        ax.set_title("Position error (m)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position error (m)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=7)

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig, axes
