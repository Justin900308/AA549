"""Plotting utilities for flat-earth navigation comparison."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_case(result: dict, save_path: str | None = None):
    t = result["time"]
    true_p = result["true_p"]
    lms = result["landmarks"]

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(true_p[:, 0], true_p[:, 1], true_p[:, 2], label="true", linewidth=2)
    ax3d.plot(result["ekf_p"][:, 0], result["ekf_p"][:, 1], result["ekf_p"][:, 2], "--", label="EKF")
    ax3d.plot(result["liekf_p"][:, 0], result["liekf_p"][:, 1], result["liekf_p"][:, 2], "-", label="LIEKF")
    ax3d.plot(result["iscvx_p"][:, 0], result["iscvx_p"][:, 1], result["iscvx_p"][:, 2], ":", linewidth=2.5, label="ISCVX-EKF")
    ax3d.scatter(lms[:, 0], lms[:, 1], lms[:, 2], marker="o", s=45, label="landmarks")
    ax3d.scatter(lms[:, 0], lms[:, 1], np.zeros(len(lms)), marker="x", s=35, label="landmark projections")
    ax3d.set_title(f"Trajectory, {result['case_name']}")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, result["ekf_att_err_deg"], "--", label="EKF")
    ax.plot(t, result["liekf_att_err_deg"], "-", label="LIEKF")
    ax.plot(t, result["iscvx_att_err_deg"], ":", linewidth=2.2, label="ISCVX-EKF")
    ax.set_title("Attitude error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [deg]")
    ax.grid(True)
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, result["ekf_pos_err_m"], "--", label="EKF")
    ax.plot(t, result["liekf_pos_err_m"], "-", label="LIEKF")
    ax.plot(t, result["iscvx_pos_err_m"], ":", linewidth=2.2, label="ISCVX-EKF")
    ax.set_title("Position error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m]")
    ax.grid(True)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_all_cases(results: list[dict], save_path: str | None = None):
    fig, axes = plt.subplots(2, len(results), figsize=(6.2 * len(results), 7.0), constrained_layout=True)
    if len(results) == 1:
        axes = axes.reshape(2, 1)
    for j, res in enumerate(results):
        t = res["time"]
        axes[0, j].plot(t, res["ekf_att_err_deg"], "--", label="EKF")
        axes[0, j].plot(t, res["liekf_att_err_deg"], "-", label="LIEKF")
        axes[0, j].plot(t, res["iscvx_att_err_deg"], ":", linewidth=2.2, label="ISCVX-EKF")
        axes[0, j].set_title(f"{res['case_name']}: attitude")
        axes[0, j].set_xlabel("time [s]")
        axes[0, j].set_ylabel("deg")
        axes[0, j].grid(True)
        axes[0, j].legend()

        axes[1, j].plot(t, res["ekf_pos_err_m"], "--", label="EKF")
        axes[1, j].plot(t, res["liekf_pos_err_m"], "-", label="LIEKF")
        axes[1, j].plot(t, res["iscvx_pos_err_m"], ":", linewidth=2.2, label="ISCVX-EKF")
        axes[1, j].set_title(f"{res['case_name']}: position")
        axes[1, j].set_xlabel("time [s]")
        axes[1, j].set_ylabel("m")
        axes[1, j].grid(True)
        axes[1, j].legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
