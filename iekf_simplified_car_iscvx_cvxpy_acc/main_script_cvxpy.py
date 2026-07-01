"""Replicate the simplified-car EKF/LIEKF/ISCVX example from Barrau-Bonnabel TAC 2017.

Run:
    python main_script.py

The setup matches Sec. IV-D of the paper:
  * unicycle/simplified-car dynamics,
  * 10 m diameter circle,
  * 40 s simulation,
  * odometer/differential odometry at 100 Hz,
  * GPS position measurement at 1 Hz,
  * N = I_2,
  * Q = diag((pi/180)^2, 1e-4, 1e-4),
  * two initial heading errors: 1 deg and 45 deg,
  * initial position known.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from continuous_discrete_ekf import ContinuousDiscreteCarEKF
from continuous_discrete_liekf import ContinuousDiscreteCarLIEKF
from continuous_discrete_iscvx_cvxpy import ContinuousDiscreteCarISCVXCVXPY, CVXPY_AVAILABLE
from dynamics import heading_error_deg, position_error, propagate_pose_rk4, wrap_angle
from plotting import plot_simplified_car_cases


# Paper parameters.
dt = 0.01                         # 100 Hz odometry propagation
T_FINAL = 40.0                     # seconds
T = int(T_FINAL / dt) + 1
GPS_DT = 1.0                       # 1 Hz GPS
UPDATE_STRIDE = int(GPS_DT / dt)

CIRCLE_DIAMETER = 10.0             # meters
RADIUS = CIRCLE_DIAMETER / 2.0
omega_const = 2.0 * np.pi / T_FINAL
v_const = RADIUS * omega_const
Q = np.diag([(np.pi / 180.0) ** 2, 1e-4, 1e-4])
N = np.eye(2)
H0 = np.array([np.pi / 2.0, RADIUS, 0.0])  # starts on a radius-5 circle centered at origin

# The paper's Fig. 1 uses 1 degree and 45 degree initial heading errors.
INITIAL_HEADING_ERRORS_DEG = [1.0, 45.0]

# Keep actual process and measurement noise off to reproduce the deterministic observer comparison.
# Q and N are still used as EKF/LIEKF design/tuning matrices, as in the paper.
ADD_SIMULATION_NOISE = False
RNG_SEED = 13


def control_profile(_t: float) -> np.ndarray:
    return np.array([v_const, omega_const], dtype=float)


def traj_simulation(add_noise: bool = ADD_SIMULATION_NOISE):
    rng = np.random.default_rng(RNG_SEED)
    time = np.arange(T) * dt
    true = np.zeros((T, 3))
    odom = np.zeros((T - 1, 2))
    gps = np.full((T, 2), np.nan)
    update_mask = np.zeros(T, dtype=bool)
    true[0] = H0

    for k in range(T - 1):
        u_true = control_profile(time[k])
        odom[k] = u_true.copy()
        if add_noise:
            # Optional simulated sensor perturbations.  Off by default for paper-style observer tests.
            beta = rng.multivariate_normal(np.zeros(3), Q)
            odom[k, 1] += beta[0]
            odom[k, 0] += beta[1]
        true[k + 1] = propagate_pose_rk4(true[k], u_true, dt)

        if (k + 1) % UPDATE_STRIDE == 0:
            gps[k + 1] = true[k + 1, 1:3]
            if add_noise:
                gps[k + 1] += rng.multivariate_normal(np.zeros(2), N)
            update_mask[k + 1] = True

    return time, true, odom, gps, update_mask


def run_case(initial_heading_error_deg: float):
    time, true, odom, gps, update_mask = traj_simulation()

    z0 = true[0].copy()
    z0[0] = wrap_angle(z0[0] + np.deg2rad(initial_heading_error_deg))
    # Initial position is assumed known.  A tiny epsilon keeps the covariance numerically well-conditioned.
    P0 = np.diag([np.deg2rad(initial_heading_error_deg) ** 2, 1e-12, 1e-12])

    ekf = ContinuousDiscreteCarEKF(z0=z0, P0=P0, Q=Q, N=N, dt=dt)
    liekf = ContinuousDiscreteCarLIEKF(z0=z0, P0=P0, Q=Q, N=N, dt=dt)
    iscvx = ContinuousDiscreteCarISCVXCVXPY(
        z0=z0,
        P0=P0,
        Q=Q,
        N=N,
        dt=dt,
        trust_radius=0.5,
        max_scp_iters=5,
        solver=None,
        fallback_without_cvxpy=True,
    )

    z_ekf = np.zeros_like(true)
    z_liekf = np.zeros_like(true)
    z_iscvx = np.zeros_like(true)
    z_ekf[0] = ekf.z
    z_liekf[0] = liekf.z
    z_iscvx[0] = iscvx.z

    for k in range(T - 1):
        yk = gps[k + 1] if update_mask[k + 1] else None
        z_ekf[k + 1] = ekf.step(odom[k], yk)
        z_liekf[k + 1] = liekf.step(odom[k], yk)
        z_iscvx[k + 1] = iscvx.step(odom[k], yk)

    heading_ekf = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, z_ekf)])
    heading_liekf = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, z_liekf)])
    heading_iscvx = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, z_iscvx)])
    pos_ekf = np.array([position_error(zt, zh) for zt, zh in zip(true, z_ekf)])
    pos_liekf = np.array([position_error(zt, zh) for zt, zh in zip(true, z_liekf)])
    pos_iscvx = np.array([position_error(zt, zh) for zt, zh in zip(true, z_iscvx)])

    return {
        "initial_heading_error_deg": float(initial_heading_error_deg),
        "time": time,
        "true": true,
        "ekf": z_ekf,
        "liekf": z_liekf,
        "iscvx": z_iscvx,
        "gps": gps,
        "update_mask": update_mask,
        "heading_error_ekf_deg": heading_ekf,
        "heading_error_liekf_deg": heading_liekf,
        "heading_error_iscvx_deg": heading_iscvx,
        "position_error_ekf_m": pos_ekf,
        "position_error_liekf_m": pos_liekf,
        "position_error_iscvx_m": pos_iscvx,
        "ekf_heading_rmse_deg": float(np.sqrt(np.mean(heading_ekf**2))),
        "liekf_heading_rmse_deg": float(np.sqrt(np.mean(heading_liekf**2))),
        "iscvx_heading_rmse_deg": float(np.sqrt(np.mean(heading_iscvx**2))),
        "ekf_position_rmse_m": float(np.sqrt(np.mean(pos_ekf**2))),
        "liekf_position_rmse_m": float(np.sqrt(np.mean(pos_liekf**2))),
        "iscvx_position_rmse_m": float(np.sqrt(np.mean(pos_iscvx**2))),
        "ekf_final_heading_error_deg": float(heading_ekf[-1]),
        "liekf_final_heading_error_deg": float(heading_liekf[-1]),
        "iscvx_final_heading_error_deg": float(heading_iscvx[-1]),
        "ekf_final_position_error_m": float(pos_ekf[-1]),
        "liekf_final_position_error_m": float(pos_liekf[-1]),
        "iscvx_final_position_error_m": float(pos_iscvx[-1]),
        "iscvx_used_cvxpy": bool(iscvx.used_cvxpy),
        "iscvx_used_fallback": bool(iscvx.used_fallback),
    }


def Estimator_sim():
    return [run_case(err) for err in INITIAL_HEADING_ERRORS_DEG]


if __name__ == "__main__":
    results = Estimator_sim()
    print("Simplified-car paper replication parameters")
    print(f"  dt = {dt:.3f} s, odometry rate = {1/dt:.0f} Hz")
    print(f"  GPS update period = {UPDATE_STRIDE} dt = {GPS_DT:.1f} s")
    print(f"  circle diameter = {CIRCLE_DIAMETER:.1f} m, final time = {T_FINAL:.1f} s")
    print(f"  v = {v_const:.6f} m/s, omega = {omega_const:.6f} rad/s")
    print(f"  Q = diag({Q[0,0]:.8e}, {Q[1,1]:.1e}, {Q[2,2]:.1e})")
    print("  N = I_2")
    print(f"  CVXPY available: {CVXPY_AVAILABLE}")
    if not CVXPY_AVAILABLE:
        print("  CVXPY backend requested, but cvxpy is not installed here; direct QP fallback is enabled.")
    print()

    summary = []
    for res in results:
        summary.append(
            {
                "initial_heading_error_deg": res["initial_heading_error_deg"],
                "ekf_heading_rmse_deg": res["ekf_heading_rmse_deg"],
                "liekf_heading_rmse_deg": res["liekf_heading_rmse_deg"],
                "iscvx_heading_rmse_deg": res["iscvx_heading_rmse_deg"],
                "ekf_position_rmse_m": res["ekf_position_rmse_m"],
                "liekf_position_rmse_m": res["liekf_position_rmse_m"],
                "iscvx_position_rmse_m": res["iscvx_position_rmse_m"],
                "ekf_final_heading_error_deg": res["ekf_final_heading_error_deg"],
                "liekf_final_heading_error_deg": res["liekf_final_heading_error_deg"],
                "iscvx_final_heading_error_deg": res["iscvx_final_heading_error_deg"],
                "ekf_final_position_error_m": res["ekf_final_position_error_m"],
                "liekf_final_position_error_m": res["liekf_final_position_error_m"],
                "iscvx_final_position_error_m": res["iscvx_final_position_error_m"],
                "iscvx_used_cvxpy": res["iscvx_used_cvxpy"],
                "iscvx_used_fallback": res["iscvx_used_fallback"],
            }
        )
        print(f"Initial heading error: {res['initial_heading_error_deg']:.0f} deg")
        print(f"  EKF   heading RMSE: {res['ekf_heading_rmse_deg']:.4f} deg")
        print(f"  LIEKF heading RMSE: {res['liekf_heading_rmse_deg']:.4f} deg")
        print(f"  ISCVX heading RMSE: {res['iscvx_heading_rmse_deg']:.4f} deg")
        print(f"  EKF   position RMSE: {res['ekf_position_rmse_m']:.4f} m")
        print(f"  LIEKF position RMSE: {res['liekf_position_rmse_m']:.4f} m")
        print(f"  ISCVX position RMSE: {res['iscvx_position_rmse_m']:.4f} m")
        print(f"  EKF   final errors: heading={res['ekf_final_heading_error_deg']:.4f} deg, pos={res['ekf_final_position_error_m']:.4f} m")
        print(f"  LIEKF final errors: heading={res['liekf_final_heading_error_deg']:.4f} deg, pos={res['liekf_final_position_error_m']:.4f} m")
        print(f"  ISCVX final errors: heading={res['iscvx_final_heading_error_deg']:.4f} deg, pos={res['iscvx_final_position_error_m']:.4f} m")
        print()

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_simplified_car_cases(results, save_path=str(out_dir / "simplified_car_ekf_liekf_iscvx_cvxpy_comparison.png"))
    plt.show()
