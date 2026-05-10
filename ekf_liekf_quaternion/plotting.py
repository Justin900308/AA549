"""Plotting helpers for the quaternion EKF/LIEKF comparison."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dynamics import attitude_error_angle, quat_rotate


def NED_2_plot_frame(v):
    """Optional display conversion used by the original visualization."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([v[1], v[0], -v[2]])


def plotting3d(q_traj, omega_traj=None, T=None, pause: float = 0.01):
    """Animate the body axes. This keeps the spirit of the original file."""
    if T is None:
        T = len(q_traj)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    axes = [(np.array([1.0, 0.0, 0.0]), "r"),
            (np.array([0.0, 1.0, 0.0]), "g"),
            (np.array([0.0, 0.0, 1.0]), "b")]

    for t in range(T):
        qt = q_traj[t]
        for axis, color in axes:
            body_axis_t = NED_2_plot_frame(quat_rotate(qt, axis))
            body_axis_t = body_axis_t / np.linalg.norm(body_axis_t)
            ax.plot([0, body_axis_t[0]], [0, body_axis_t[1]], [0, body_axis_t[2]], color)

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel("East")
        ax.set_ylabel("North")
        ax.set_zlabel("Up")

        if t < T - 1:
            plt.pause(pause)
            ax.clear()
        else:
            plt.show()


def plot_filter_comparison(time, q_true, q_ekf, q_liekf, update_mask=None, save_path=None):
    """Plot attitude error and quaternion estimates for the two filters."""
    time = np.asarray(time)
    ekf_err = np.array([attitude_error_angle(qt, qe) for qt, qe in zip(q_true, q_ekf)])
    liekf_err = np.array([attitude_error_angle(qt, ql) for qt, ql in zip(q_true, q_liekf)])

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].plot(time, np.rad2deg(ekf_err), label="additive EKF")
    axs[0].plot(time, np.rad2deg(liekf_err), label="LIEKF")
    if update_mask is not None:
        update_times = time[np.asarray(update_mask, dtype=bool)]
        for ut in update_times:
            axs[0].axvline(ut, linewidth=0.5, alpha=0.12)
    axs[0].set_ylabel("attitude error [deg]")
    axs[0].grid(True)
    axs[0].legend()

    labels = ["qw", "qx", "qy", "qz"]
    for i, label in enumerate(labels):
        axs[1].plot(time, q_true[:, i], "k--", linewidth=1.0, alpha=0.7 if i == 0 else 0.35)
        axs[1].plot(time, q_ekf[:, i], linewidth=0.9, alpha=0.7, label=f"EKF {label}" if i == 0 else None)
        axs[1].plot(time, q_liekf[:, i], linewidth=0.9, alpha=0.7, label=f"LIEKF {label}" if i == 0 else None)
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("quaternion components")
    axs[1].grid(True)
    axs[1].legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, axs
