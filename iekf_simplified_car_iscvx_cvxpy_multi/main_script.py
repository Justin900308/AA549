

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from continuous_discrete_ekf import ContinuousDiscreteCarEKF
from continuous_discrete_liekf import ContinuousDiscreteCarLIEKF
from continuous_discrete_particle_filter import ContinuousDiscreteCarParticleFilter
from iscvx_cvxpy_l2_backtracking import ContinuousDiscreteCarISCVXCVXPY
from dynamics import heading_error_deg, position_error, unicycle_dynamics, wrap_angle
from plotting import plot_simplified_car_cases, plot_monte_carlo_error_3sigma
from integrator import rk4
import constants as ct
from traj_gen import traj_gen_fun


#### Select case
Case = 1  # for the circle example
Case = 2  # for the constrained case

#### Monte-Carlo settings
N_TRIALS = 20
BASE_SEED = 225678  # change this to get a different Monte-Carlo experiment

def control_profile(_t: float) -> np.ndarray:
    return np.array([ct.v_const, ct.omega_const], dtype=float)


def _make_trial_rng(trial_idx: int, case_idx: int = 0) -> np.random.Generator:
    """Create an independent, reproducible RNG for each trial/case.

    Important: do not do ``rng += 100``.  ``rng`` must be an actual random
    number generator, not just a seed-like variable.  SeedSequence gives
    independent streams while keeping the entire run reproducible.
    """
    seed_seq = np.random.SeedSequence([BASE_SEED, trial_idx, case_idx])
    return np.random.default_rng(seed_seq)


def traj_simulation(
    z_traj: np.ndarray,
    u_traj: np.ndarray,
    rng: np.random.Generator,
    add_noise: bool = ct.ADD_SIMULATION_NOISE,
):
    # rng = np.random.default_rng(ct.RNG_SEED)
    time = np.arange(ct.T) * ct.dt
    true = np.zeros((ct.T, 3))
    odom = np.zeros((ct.T - 1, 2))
    gps = np.full((ct.T, 2), np.nan)
    update_mask = np.zeros(ct.T, dtype=bool)

    # Use an integer stride.  The original value is usually exactly 10, but
    # rounding avoids silent bugs from floating-point modulo operations.
    dt_ratio = int(round(ct.dt_traj_gen / ct.dt))
    control_count = 0

    for k in range(ct.T - 1):
        if Case == 1:
            u_true = control_profile(time[k])
        else:
            u_true = u_traj[control_count]

        if k % dt_ratio == 0:
            control_count += 1
            if control_count == ct.T_traj_gen - 1:
                control_count -= 1

        odom[k] = u_true.copy()
        if add_noise:
            beta = rng.multivariate_normal(np.zeros(3), ct.Q)
            odom[k, 1] += beta[0]  # angular-rate noise
            odom[k, 0] += beta[1]  # forward-speed noise

        true[k + 1] = rk4(unicycle_dynamics, true[k], u_true, ct.dt)

        if (k + 1) % ct.UPDATE_STRIDE == 0:
            gps[k + 1] = true[k + 1, 1:3]
            if add_noise:
                gps[k + 1] += rng.multivariate_normal(np.zeros(2), ct.N)
            update_mask[k + 1] = True

    return time, true, odom, gps, update_mask


def run_case(
    initial_heading_error_deg: float,
    z_traj: np.ndarray,
    u_traj: np.ndarray,
    rng: np.random.Generator,
):
    time, true, odom, gps, update_mask = traj_simulation(z_traj, u_traj, rng,True)

    z0 = true[0].copy()
    z0[0] = wrap_angle(z0[0] + np.deg2rad(initial_heading_error_deg))

    # Initial position is assumed known.  A tiny epsilon keeps the covariance
    # numerically well-conditioned.
    P0 = np.diag([np.deg2rad(initial_heading_error_deg) ** 2, 1e-12, 1e-12])

    particle = ContinuousDiscreteCarParticleFilter(z0=z0, P0=P0, Q=ct.Q, N=ct.N, dt=ct.dt, rng=rng)
    ekf = ContinuousDiscreteCarEKF(z0=z0, P0=P0, Q=ct.Q, N=ct.N, dt=ct.dt)
    liekf = ContinuousDiscreteCarLIEKF(z0=z0, P0=P0, Q=ct.Q, N=ct.N, dt=ct.dt)

    constraint_flag = False if Case == 1 else True
    iscvx = ContinuousDiscreteCarISCVXCVXPY(
        z0=z0,
        P0=P0,
        Q=ct.Q,
        N=ct.N,
        dt=ct.dt,
        flag=constraint_flag,
        max_scp_iters=5,
        obs=ct.obs,
        obs_r=ct.obs_r,
    )

    z_particle = np.zeros_like(true)
    z_ekf = np.zeros_like(true)
    z_liekf = np.zeros_like(true)
    z_iscvx = np.zeros_like(true)
    z_particle[0] = particle.z
    z_ekf[0] = ekf.z
    z_liekf[0] = liekf.z
    z_iscvx[0] = iscvx.z

    condition_times_sum = np.zeros(4)
    count = 0

    for k in range(ct.T - 1):
        yk = gps[k + 1] if update_mask[k + 1] else None

        z_particle[k + 1] = particle.step(odom[k], yk)
        z_ekf[k + 1] = ekf.step(odom[k], yk)
        z_liekf[k + 1] = liekf.step(odom[k], yk)
        z_iscvx[k + 1] = iscvx.step(odom[k], yk)

        if update_mask[k + 1] and ekf.update_t is not None:
            condition_times_sum += np.array([particle.update_t, ekf.update_t, liekf.update_t, iscvx.update_t])
            count += 1

    condition_times_avg = condition_times_sum / count if count > 0 else np.full(4, np.nan)

    heading_particle = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, z_particle)])
    heading_ekf = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, z_ekf)])
    heading_liekf = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, z_liekf)])
    heading_iscvx = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, z_iscvx)])

    pos_particle = np.array([position_error(zt, zh) for zt, zh in zip(true, z_particle)])
    pos_ekf = np.array([position_error(zt, zh) for zt, zh in zip(true, z_ekf)])
    pos_liekf = np.array([position_error(zt, zh) for zt, zh in zip(true, z_liekf)])
    pos_iscvx = np.array([position_error(zt, zh) for zt, zh in zip(true, z_iscvx)])

    return {
        "initial_heading_error_deg": float(initial_heading_error_deg),
        "time": time,
        "true": true,
        "particle": z_particle,
        "ekf": z_ekf,
        "liekf": z_liekf,
        "iscvx": z_iscvx,
        "gps": gps,
        "update_mask": update_mask,
        "heading_error_particle_deg": heading_particle,
        "heading_error_ekf_deg": heading_ekf,
        "heading_error_liekf_deg": heading_liekf,
        "heading_error_iscvx_deg": heading_iscvx,
        "position_error_particle_m": pos_particle,
        "position_error_ekf_m": pos_ekf,
        "position_error_liekf_m": pos_liekf,
        "position_error_iscvx_m": pos_iscvx,
        "particle_heading_rmse_deg": float(np.sqrt(np.mean(heading_particle**2))),
        "ekf_heading_rmse_deg": float(np.sqrt(np.mean(heading_ekf**2))),
        "liekf_heading_rmse_deg": float(np.sqrt(np.mean(heading_liekf**2))),
        "iscvx_heading_rmse_deg": float(np.sqrt(np.mean(heading_iscvx**2))),
        "particle_position_rmse_m": float(np.sqrt(np.mean(pos_particle**2))),
        "ekf_position_rmse_m": float(np.sqrt(np.mean(pos_ekf**2))),
        "liekf_position_rmse_m": float(np.sqrt(np.mean(pos_liekf**2))),
        "iscvx_position_rmse_m": float(np.sqrt(np.mean(pos_iscvx**2))),
        "particle_final_heading_error_deg": float(heading_particle[-1]),
        "ekf_final_heading_error_deg": float(heading_ekf[-1]),
        "liekf_final_heading_error_deg": float(heading_liekf[-1]),
        "iscvx_final_heading_error_deg": float(heading_iscvx[-1]),
        "particle_final_position_error_m": float(pos_particle[-1]),
        "ekf_final_position_error_m": float(pos_ekf[-1]),
        "liekf_final_position_error_m": float(pos_liekf[-1]),
        "iscvx_final_position_error_m": float(pos_iscvx[-1]),
        "iscvx_used_cvxpy": bool(iscvx.used_cvxpy),
        "iscvx_used_fallback": bool(iscvx.used_fallback),
        "condition_times": condition_times_avg,
    }


def estimator_sim(z_traj: np.ndarray, u_traj: np.ndarray, trial_idx: int):
    results = []
    for case_idx, err in enumerate(ct.INITIAL_HEADING_ERRORS_DEG):
        # rng = _make_trial_rng(trial_idx, case_idx)
        rng = np.random.default_rng(ct.RNG_SEED + trial_idx)
        results.append(run_case(err, z_traj, u_traj, rng))
    return results


def summarize_trial(results):
    summary = []
    for res in results:
        summary.append(
            {
                "initial_heading_error_deg": res["initial_heading_error_deg"],
                "particle_heading_rmse_deg": res["particle_heading_rmse_deg"],
                "ekf_heading_rmse_deg": res["ekf_heading_rmse_deg"],
                "liekf_heading_rmse_deg": res["liekf_heading_rmse_deg"],
                "iscvx_heading_rmse_deg": res["iscvx_heading_rmse_deg"],
                "particle_position_rmse_m": res["particle_position_rmse_m"],
                "ekf_position_rmse_m": res["ekf_position_rmse_m"],
                "liekf_position_rmse_m": res["liekf_position_rmse_m"],
                "iscvx_position_rmse_m": res["iscvx_position_rmse_m"],
                "particle_final_heading_error_deg": res["particle_final_heading_error_deg"],
                "ekf_final_heading_error_deg": res["ekf_final_heading_error_deg"],
                "liekf_final_heading_error_deg": res["liekf_final_heading_error_deg"],
                "iscvx_final_heading_error_deg": res["iscvx_final_heading_error_deg"],
                "particle_final_position_error_m": res["particle_final_position_error_m"],
                "ekf_final_position_error_m": res["ekf_final_position_error_m"],
                "liekf_final_position_error_m": res["liekf_final_position_error_m"],
                "iscvx_final_position_error_m": res["iscvx_final_position_error_m"],
                "iscvx_used_cvxpy": res["iscvx_used_cvxpy"],
                "iscvx_used_fallback": res["iscvx_used_fallback"],
            }
        )
    return summary


def print_trial_summary(trial_idx: int, results):
    print(f"\nMonte-Carlo trial {trial_idx + 1}")
    for res in results:
        print(f"Initial heading error: {res['initial_heading_error_deg']:.0f} deg")
        print(f"  PF    heading RMSE: {res['particle_heading_rmse_deg']:.4f} deg")
        print(f"  EKF   heading RMSE: {res['ekf_heading_rmse_deg']:.4f} deg")
        print(f"  LIEKF heading RMSE: {res['liekf_heading_rmse_deg']:.4f} deg")
        print(f"  ISCVX heading RMSE: {res['iscvx_heading_rmse_deg']:.4f} deg")
        print(f"  PF    position RMSE: {res['particle_position_rmse_m']:.4f} m")
        print(f"  EKF   position RMSE: {res['ekf_position_rmse_m']:.4f} m")
        print(f"  LIEKF position RMSE: {res['liekf_position_rmse_m']:.4f} m")
        print(f"  ISCVX position RMSE: {res['iscvx_position_rmse_m']:.4f} m")
        print(
            "  avg update time [PF, EKF, LIEKF, ISCVX]: "
            f"{np.array2string(res['condition_times'], precision=5)} s"
        )


def main():
    z_traj = np.zeros([ct.T_traj_gen, ct.n])
    z_traj[0] = ct.z_0
    u_traj = np.zeros([ct.T_traj_gen - 1, ct.m])

    if Case == 2:
        z_traj, u_traj, _ = traj_gen_fun(z_traj, u_traj)

    print("Simplified-car paper replication parameters")
    print(f"  Monte-Carlo trials = {N_TRIALS}")
    print(f"  base seed = {BASE_SEED}")
    print(f"  dt = {ct.dt:.3f} s, odometry rate = {1 / ct.dt:.0f} Hz")
    print(f"  GPS update period = {ct.UPDATE_STRIDE} dt = {ct.GPS_DT:.1f} s")
    print(f"  circle diameter = {ct.CIRCLE_DIAMETER:.1f} m, final time = {ct.T_FINAL:.1f} s")
    print(f"  v = {ct.v_const:.6f} m/s, omega = {ct.omega_const:.6f} rad/s")
    print(f"  Q = diag({ct.Q[0, 0]:.8e}, {ct.Q[1, 1]:.1e}, {ct.Q[2, 2]:.1e})")
    print("  N = I_2")

    all_trials = []
    all_summaries = []
    for trial_idx in range(N_TRIALS):
        results = estimator_sim(z_traj, u_traj, trial_idx)
        all_trials.append(results)
        all_summaries.append({"trial": trial_idx, "results": summarize_trial(results)})
        print_trial_summary(trial_idx, results)

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "summary_monte_carlo.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    # Keep the original single-trial figure for trajectory/error traces.
    plot_simplified_car_cases(
        Case,
        all_trials[0],
        save_path=str(out_dir / "simplified_car_single_trial_comparison.png"),
    )

    # New Monte-Carlo figure: mean error over N trials with empirical +/- 3 sigma bands.
    plot_monte_carlo_error_3sigma(
        Case,
        all_trials,
        save_path=str(out_dir / "simplified_car_monte_carlo_error_3sigma.png"),
    )

    plt.show()


if __name__ == "__main__":
    main()
