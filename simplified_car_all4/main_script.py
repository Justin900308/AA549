"""Simplified-car comparison: EKF, LIEKF, ISCVX, and RIEKF.

Run:
    python main_script.py

This script keeps the paper's simplified-car GPS setup for the EKF/LIEKF and
ISCVX methods.  It also adds the paper's RIEKF, which is defined for the
right-invariant landmark/range-bearing output, not for the GPS output.

Therefore the simulation generates two discrete outputs at 1 Hz:

    GPS position:          y_gps = x + V,
    landmark body vectors: y_k   = R(theta)^T (x - p_k) + V_k.

The RIEKF uses the landmark output.  The other three filters use GPS, matching
the earlier comparison and the paper's LIEKF/GPS example.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from continuous_discrete_ekf import ContinuousDiscreteCarEKF
from continuous_discrete_liekf import ContinuousDiscreteCarLIEKF
from continuous_discrete_iscvx_cvxpy import ContinuousDiscreteCarISCVXCVXPY, CVXPY_AVAILABLE
from continuous_discrete_riekf import ContinuousDiscreteCarRIEKF
from dynamics import (
    heading_error_deg,
    landmark_body_measurement_model,
    position_error,
    propagate_pose_rk4,
    wrap_angle,
)
from plotting import plot_simplified_car_cases


# -----------------------------------------------------------------------------
# Paper parameters for the simplified-car example.
# -----------------------------------------------------------------------------

dt = 0.01                         # 100 Hz odometry propagation
T_FINAL = 40.0                     # seconds
T = int(T_FINAL / dt) + 1
GPS_DT = 1.0                       # 1 Hz discrete updates
UPDATE_STRIDE = int(GPS_DT / dt)

CIRCLE_DIAMETER = 10.0             # meters
RADIUS = CIRCLE_DIAMETER / 2.0
omega_const = 2.0 * np.pi / T_FINAL
v_const = RADIUS * omega_const

# Paper tuning matrices for odometry propagation and GPS.
Q = np.diag([(np.pi / 180.0) ** 2, 1e-4, 1e-4])
N_GPS = np.eye(2)

# Landmark covariance for the RIEKF.  Using I_2 makes it comparable to N=I_2.
N_LANDMARK_EACH = np.eye(2)

# Two distinct landmarks are enough for the paper's RIEKF observability result.
LANDMARKS = np.array(
    [
        [8.0, -2.0],
        [-4.0, 7.0],
        [-6.0, -5.0],
        [9.0, 6.0],
    ],
    dtype=float,
)

H0 = np.array([np.pi / 2.0, RADIUS, 0.0])  # starts on a radius-5 circle
INITIAL_HEADING_ERRORS_DEG = [1.0, 45.0]

# Noise is off by default to reproduce the deterministic observer comparison.
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
    landmarks_y = np.full((T, 2 * len(LANDMARKS)), np.nan)
    update_mask = np.zeros(T, dtype=bool)
    true[0] = H0

    for k in range(T - 1):
        u_true = control_profile(time[k])
        odom[k] = u_true.copy()
        if add_noise:
            beta = rng.multivariate_normal(np.zeros(3), Q)
            odom[k, 1] += beta[0]  # angular perturbation
            odom[k, 0] += beta[1]  # longitudinal perturbation

        true[k + 1] = propagate_pose_rk4(true[k], u_true, dt)

        if (k + 1) % UPDATE_STRIDE == 0:
            gps[k + 1] = true[k + 1, 1:3]
            landmarks_y[k + 1] = landmark_body_measurement_model(true[k + 1], LANDMARKS)
            if add_noise:
                gps[k + 1] += rng.multivariate_normal(np.zeros(2), N_GPS)
                noise_blocks = [
                    rng.multivariate_normal(np.zeros(2), N_LANDMARK_EACH)
                    for _ in range(len(LANDMARKS))
                ]
                landmarks_y[k + 1] += np.concatenate(noise_blocks)
            update_mask[k + 1] = True

    return time, true, odom, gps, landmarks_y, update_mask


def _series_errors(true, est):
    heading = np.array([heading_error_deg(zt, zh) for zt, zh in zip(true, est)])
    pos = np.array([position_error(zt, zh) for zt, zh in zip(true, est)])
    return heading, pos


def run_case(initial_heading_error_deg: float):
    time, true, odom, gps, landmarks_y, update_mask = traj_simulation()

    z0 = true[0].copy()
    z0[0] = wrap_angle(z0[0] + np.deg2rad(initial_heading_error_deg))

    # The paper assumes the initial position is known and the covariance encodes
    # the initial heading error standard deviation.
    P0 = np.diag([np.deg2rad(initial_heading_error_deg) ** 2, 1e-12, 1e-12])

    ekf = ContinuousDiscreteCarEKF(z0=z0, P0=P0, Q=Q, N=N_GPS, dt=dt)
    liekf = ContinuousDiscreteCarLIEKF(z0=z0, P0=P0, Q=Q, N=N_GPS, dt=dt)
    iscvx = ContinuousDiscreteCarISCVXCVXPY(
        z0=z0,
        P0=P0,
        Q=Q,
        N=N_GPS,
        dt=dt,
        trust_radius=0.5,
        max_scp_iters=5,
        solver=None,
    )
    riekf = ContinuousDiscreteCarRIEKF(
        z0=z0,
        P0=P0,
        Q=Q,
        N_each=N_LANDMARK_EACH,
        landmarks=LANDMARKS,
        dt=dt,
    )

    z_ekf = np.zeros_like(true)
    z_liekf = np.zeros_like(true)
    z_iscvx = np.zeros_like(true)
    z_riekf = np.zeros_like(true)
    z_ekf[0] = ekf.z
    z_liekf[0] = liekf.z
    z_iscvx[0] = iscvx.z
    z_riekf[0] = riekf.z

    for k in range(T - 1):
        y_gps = gps[k + 1] if update_mask[k + 1] else None
        y_lm = landmarks_y[k + 1] if update_mask[k + 1] else None

        z_ekf[k + 1] = ekf.step(odom[k], y_gps)
        z_liekf[k + 1] = liekf.step(odom[k], y_gps)
        z_iscvx[k + 1] = iscvx.step(odom[k], y_gps)
        z_riekf[k + 1] = riekf.step(odom[k], y_lm)

    heading_ekf, pos_ekf = _series_errors(true, z_ekf)
    heading_liekf, pos_liekf = _series_errors(true, z_liekf)
    heading_iscvx, pos_iscvx = _series_errors(true, z_iscvx)
    heading_riekf, pos_riekf = _series_errors(true, z_riekf)

    return {
        "initial_heading_error_deg": float(initial_heading_error_deg),
        "time": time,
        "true": true,
        "ekf": z_ekf,
        "liekf": z_liekf,
        "iscvx": z_iscvx,
        "riekf": z_riekf,
        "gps": gps,
        "landmark_measurements": landmarks_y,
        "landmarks": LANDMARKS,
        "update_mask": update_mask,
        "heading_error_ekf_deg": heading_ekf,
        "heading_error_liekf_deg": heading_liekf,
        "heading_error_iscvx_deg": heading_iscvx,
        "heading_error_riekf_deg": heading_riekf,
        "position_error_ekf_m": pos_ekf,
        "position_error_liekf_m": pos_liekf,
        "position_error_iscvx_m": pos_iscvx,
        "position_error_riekf_m": pos_riekf,
        "ekf_heading_rmse_deg": float(np.sqrt(np.mean(heading_ekf**2))),
        "liekf_heading_rmse_deg": float(np.sqrt(np.mean(heading_liekf**2))),
        "iscvx_heading_rmse_deg": float(np.sqrt(np.mean(heading_iscvx**2))),
        "riekf_heading_rmse_deg": float(np.sqrt(np.mean(heading_riekf**2))),
        "ekf_position_rmse_m": float(np.sqrt(np.mean(pos_ekf**2))),
        "liekf_position_rmse_m": float(np.sqrt(np.mean(pos_liekf**2))),
        "iscvx_position_rmse_m": float(np.sqrt(np.mean(pos_iscvx**2))),
        "riekf_position_rmse_m": float(np.sqrt(np.mean(pos_riekf**2))),
        "ekf_final_heading_error_deg": float(heading_ekf[-1]),
        "liekf_final_heading_error_deg": float(heading_liekf[-1]),
        "iscvx_final_heading_error_deg": float(heading_iscvx[-1]),
        "riekf_final_heading_error_deg": float(heading_riekf[-1]),
        "ekf_final_position_error_m": float(pos_ekf[-1]),
        "liekf_final_position_error_m": float(pos_liekf[-1]),
        "iscvx_final_position_error_m": float(pos_iscvx[-1]),
        "riekf_final_position_error_m": float(pos_riekf[-1]),
        "iscvx_used_cvxpy": bool(iscvx.used_cvxpy),
    }


def Estimator_sim():
    return [run_case(err) for err in INITIAL_HEADING_ERRORS_DEG]


def _summary_entry(res):
    return {
        "initial_heading_error_deg": res["initial_heading_error_deg"],
        "ekf_heading_rmse_deg": res["ekf_heading_rmse_deg"],
        "liekf_heading_rmse_deg": res["liekf_heading_rmse_deg"],
        "iscvx_heading_rmse_deg": res["iscvx_heading_rmse_deg"],
        "riekf_heading_rmse_deg": res["riekf_heading_rmse_deg"],
        "ekf_position_rmse_m": res["ekf_position_rmse_m"],
        "liekf_position_rmse_m": res["liekf_position_rmse_m"],
        "iscvx_position_rmse_m": res["iscvx_position_rmse_m"],
        "riekf_position_rmse_m": res["riekf_position_rmse_m"],
        "ekf_final_heading_error_deg": res["ekf_final_heading_error_deg"],
        "liekf_final_heading_error_deg": res["liekf_final_heading_error_deg"],
        "iscvx_final_heading_error_deg": res["iscvx_final_heading_error_deg"],
        "riekf_final_heading_error_deg": res["riekf_final_heading_error_deg"],
        "ekf_final_position_error_m": res["ekf_final_position_error_m"],
        "liekf_final_position_error_m": res["liekf_final_position_error_m"],
        "iscvx_final_position_error_m": res["iscvx_final_position_error_m"],
        "riekf_final_position_error_m": res["riekf_final_position_error_m"],
        "iscvx_used_cvxpy": res["iscvx_used_cvxpy"],
    }


if __name__ == "__main__":
    results = Estimator_sim()
    print("Simplified-car EKF/LIEKF/ISCVX/RIEKF comparison")
    print(f"  CVXPY available = {CVXPY_AVAILABLE}")
    print(f"  dt = {dt:.3f} s, odometry rate = {1/dt:.0f} Hz")
    print(f"  update period = {UPDATE_STRIDE} dt = {GPS_DT:.1f} s")
    print(f"  circle diameter = {CIRCLE_DIAMETER:.1f} m, final time = {T_FINAL:.1f} s")
    print(f"  v = {v_const:.6f} m/s, omega = {omega_const:.6f} rad/s")
    print(f"  Q = diag({Q[0,0]:.8e}, {Q[1,1]:.1e}, {Q[2,2]:.1e})")
    print("  GPS N = I_2; RIEKF landmark N_each = I_2")
    print(f"  RIEKF landmarks = {LANDMARKS.tolist()}")
    print()

    summary = []
    for res in results:
        summary.append(_summary_entry(res))
        print(f"Initial heading error: {res['initial_heading_error_deg']:.0f} deg")
        for name in ["ekf", "liekf", "iscvx", "riekf"]:
            label = name.upper()
            print(
                f"  {label:5s} heading RMSE: {res[name + '_heading_rmse_deg']:.4f} deg, "
                f"position RMSE: {res[name + '_position_rmse_m']:.4f} m, "
                f"final heading: {res[name + '_final_heading_error_deg']:.4f} deg, "
                f"final pos: {res[name + '_final_position_error_m']:.4f} m"
            )
        print()

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_simplified_car_cases(
        results,
        save_path=str(out_dir / "simplified_car_all4_comparison.png"),
    )
    plt.show()
