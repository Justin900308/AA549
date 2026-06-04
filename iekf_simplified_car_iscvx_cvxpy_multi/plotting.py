"""Plot helpers for the simplified-car replication."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import constants as ct
from matplotlib.patches import Circle


ESTIMATOR_LABELS = {
    "ekf": "EKF",
    "liekf": "LIEKF",
    "iscvx": "ISCVX",
}


def plot_simplified_car_cases(case, results_by_case, save_path: str | None = None):
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
        iscvx = res["iscvx"]

        ax = axes[0, j]
        ax.plot(true[:, 1], true[:, 2], "-", label="True trajectory")
        ax.plot(true[0, 1], true[0, 2], "r.", markersize=10, label="Initial position")
        ax.plot(true[-1, 1], true[-1, 2], "g.", markersize=10, label="Final position")
        ax.plot(ekf[:, 1], ekf[:, 2], "--", label="EKF estimate")
        ax.plot(liekf[:, 1], liekf[:, 2], "-", label="LIEKF estimate")
        ax.plot(iscvx[:, 1], iscvx[:, 2], ":", linewidth=2.0, label="ISCVX estimate")
        if case == 2:
            for i in range(ct.num_obs):
                circle = Circle((ct.obs[i, 0], ct.obs[i, 1]), ct.obs_r, fill=False, linewidth=2)
                ax.add_patch(circle)
        ax.set_title(f"$e_\\theta(0)={res['initial_heading_error_deg']:g}^\\circ$")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax = axes[1, j]
        ax.plot(t, res["heading_error_ekf_deg"], "--", label="EKF")
        ax.plot(t, res["heading_error_liekf_deg"], "-", label="LIEKF")
        ax.plot(t, res["heading_error_iscvx_deg"], ":", linewidth=2.0, label="ISCVX")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("heading error (deg)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax = axes[2, j]
        ax.plot(t, res["position_error_ekf_m"], "--", label="EKF")
        ax.plot(t, res["position_error_liekf_m"], "-", label="LIEKF")
        ax.plot(t, res["position_error_iscvx_m"], ":", linewidth=2.0, label="ISCVX")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position error (m)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig, axes


def _stack_trial_errors(all_trials, case_idx: int, error_key: str) -> np.ndarray:
    """Return array with shape (n_trials, T) for a given case/error key."""
    return np.stack([trial_results[case_idx][error_key] for trial_results in all_trials], axis=0)


def _plot_error_with_3sigma(ax, t, errors, label: str):
    """Plot empirical mean error and +/- 3 sigma bounds over trials.

    errors has shape (n_trials, T).  The line is
        mean(e(t))
    estimated over Monte-Carlo trials.  The shaded band is
        mean(e(t)) +/- 3 std(e(t)).

    This is different from plotting MSE: the plotted quantity keeps the sign
    of the heading error, and for position error it stays in meters rather than
    square meters.
    """
    mean_error = np.mean(errors, axis=0)
    ddof = 1 if errors.shape[0] > 1 else 0
    sigma = np.std(errors, axis=0, ddof=ddof)

    line = ax.plot(t, mean_error, label=f"{label} mean error")[0]
    lower = mean_error - 3.0 * sigma
    upper = mean_error + 3.0 * sigma
    ax.fill_between(t, lower, upper, alpha=0.35, color=line.get_color(), linewidth=0.0)

    return mean_error, lower, upper


def plot_monte_carlo_error_3sigma(case, all_trials, save_path: str | None = None):
    """Plot estimation error over all Monte-Carlo trials with empirical 3 sigma bands.

    Parameters
    ----------
    case:
        Same case flag used by the original plotting function.  It is kept for
        a consistent interface; this aggregate error plot itself does not need
        the obstacle geometry.
    all_trials:
        List of trial outputs.  Each element is the list returned by
        ``estimator_sim`` / ``Estimator_sim`` for all initial-heading-error
        cases.
    save_path:
        Optional output image path.
    """
    if len(all_trials) == 0:
        raise ValueError("all_trials must contain at least one Monte-Carlo trial")

    n_trials = len(all_trials)
    n_cases = len(all_trials[0])
    fig, axes = plt.subplots(2, n_cases, figsize=(6.4 * n_cases, 7.2), constrained_layout=True)
    if n_cases == 1:
        axes = axes.reshape(2, 1)

    for case_idx in range(n_cases):
        res0 = all_trials[0][case_idx]
        t = res0["time"]
        init_deg = res0["initial_heading_error_deg"]

        ax = axes[0, case_idx]
        for key, label in ESTIMATOR_LABELS.items():
            errors = _stack_trial_errors(all_trials, case_idx, f"heading_error_{key}_deg")
            _plot_error_with_3sigma(ax, t, errors, label)
        ax.axhline(0.0, linewidth=0.8)
        # ax.set_title(f"Heading error, $e_\theta(0)={init_deg:g}^\circ$, N={n_trials}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("heading error (deg)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        ax = axes[1, case_idx]
        for key, label in ESTIMATOR_LABELS.items():
            errors = _stack_trial_errors(all_trials, case_idx, f"position_error_{key}_m")
            _plot_error_with_3sigma(ax, t, errors, label)
        ax.axhline(0.0, linewidth=0.8)
        # ax.set_title(f"Position error, $e_\theta(0)={init_deg:g}^\circ$, N={n_trials}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position error (m)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig, axes
