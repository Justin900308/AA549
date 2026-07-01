"""Flat-earth navigation example: EKF vs LIEKF vs ISCVX-EKF.

This reproduces the structure of Barrau--Bonnabel's second simulation case:
orientation, velocity, and position are estimated from high-rate inertial
measurements and low-rate landmark relative-position observations.

Run:
    python main_script.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from continuous_discrete_ekf import ContinuousDiscreteNavEKF
from continuous_discrete_liekf import ContinuousDiscreteNavLIEKF
from continuous_discrete_iscvx import ContinuousDiscreteNavISCVX, CVXPY_AVAILABLE
from dynamics import (
    GRAVITY,
    LANDMARKS,
    IMUInput,
    State,
    attitude_error_deg,
    measurement_model,
    position_error,
    propagate_state,
    so3_exp,
    skew,
)
from plotting import plot_all_cases, plot_case


# Paper/benchmark parameters.
T_FINAL = 30.0
IMU_FREQ = 100
OBS_FREQ = 1
DT = 1.0 / IMU_FREQ
N_STEPS = int(T_FINAL * IMU_FREQ)  # states are 0,...,N_STEPS-1, matching UKFM style
OBS_STRIDE = int(IMU_FREQ / OBS_FREQ)
RADIUS = 5.0
OBS_STD = 0.1
OBS_COV = OBS_STD**2 * np.eye(3)
N_MEAS = OBS_STD**2 * np.eye(3 * len(LANDMARKS))

# Paper Fig. 2 tuning matrices.
Q1 = np.zeros((9, 9))
Q1[0:3, 0:3] = 1e-8 * np.eye(3)
Q1[3:6, 3:6] = 1e-8 * np.eye(3)
Q2 = np.zeros((9, 9))
Q2[0:3, 0:3] = 1e-4 * np.eye(3)
Q2[3:6, 3:6] = 1e-4 * np.eye(3)

ROT0_STD = 15.0 / np.sqrt(3.0) * np.pi / 180.0
P0_STD = 1.0 / np.sqrt(3.0)
P0_STD_COV = np.zeros((9, 9))
P0_STD_COV[0:3, 0:3] = ROT0_STD**2 * np.eye(3)
P0_STD_COV[6:9, 6:9] = P0_STD**2 * np.eye(3)

# Deterministic setting in the paper's Fig. 2 discussion.  The design matrices
# Q and N are still used by the filters, but no random noise is injected.
ADD_SIMULATION_NOISE = False
RNG_SEED = 4


def generate_true_trajectory(add_noise: bool = ADD_SIMULATION_NOISE):
    """Generate true circular trajectory and IMU/landmark data.

    This follows the open-source UKFM benchmark model: p(t) is a radius-5 circle
    over 30 seconds, R_true is identity, gyro is zero, and the accelerometer is
    chosen so that v_dot = g + R acc realizes the circular motion.
    """
    rng = np.random.default_rng(RNG_SEED)
    time = np.linspace(0.0, T_FINAL, N_STEPS)
    p = RADIUS * np.vstack(
        [
            np.sin(time / T_FINAL * 2.0 * np.pi),
            np.cos(time / T_FINAL * 2.0 * np.pi),
            np.zeros(N_STEPS),
        ]
    ).T
    v = np.vstack([np.zeros(3), np.diff(p, axis=0)]) / DT
    acc_world = np.vstack([np.zeros(3), np.diff(v, axis=0)]) / DT

    true_states: list[State] = [State(R=np.eye(3), v=v[0], p=p[0])]
    imu_inputs: list[IMUInput] = []
    for k in range(1, N_STEPS):
        gyro = np.zeros(3)
        acc_body = true_states[k - 1].R.T @ (acc_world[k - 1] - GRAVITY)
        if add_noise:
            gyro = gyro + np.sqrt(Q1[0, 0]) * rng.standard_normal(3)
            acc_body = acc_body + np.sqrt(Q1[3, 3]) * rng.standard_normal(3)
        imu_inputs.append(IMUInput(gyro=gyro, acc=acc_body))
        true_states.append(propagate_state(true_states[k - 1], imu_inputs[-1], DT, GRAVITY))

    measurements = np.full((N_STEPS, 3 * len(LANDMARKS)), np.nan)
    update_mask = np.zeros(N_STEPS, dtype=bool)
    for k in range(N_STEPS):
        if k > 0 and k % OBS_STRIDE == 0:
            y = measurement_model(true_states[k], LANDMARKS)
            if add_noise:
                y = y + OBS_STD * rng.standard_normal(3 * len(LANDMARKS))
            measurements[k] = y
            update_mask[k] = True

    return time, true_states, imu_inputs, measurements, update_mask


def initialize_estimate(true0: State) -> State:
    """Initial error: 15 deg attitude norm and 1 m position norm, no velocity error."""
    phi0 = ROT0_STD * np.array([1.0, -1.0, 1.0])
    p_err0 = P0_STD * np.array([1.0, -1.0, 1.0])
    return State(R=so3_exp(phi0) @ true0.R, v=true0.v.copy(), p=true0.p + p_err0)


def run_case(case_name: str, Q_design: np.ndarray):
    time, true_states, imu_inputs, measurements, update_mask = generate_true_trajectory()
    state0 = initialize_estimate(true_states[0])

    # The EKF uses the standard SO(3) x R^6 covariance.  The LIEKF/ISCVX use
    # right-invariant SE_2(3) covariance.  This simple J-turn matches the UKFM
    # benchmark conversion for right-invariant initialization.
    J_right = np.eye(9)
    J_right[6:9, 0:3] = skew(state0.p)
    P0_right = J_right @ P0_STD_COV @ J_right.T

    ekf = ContinuousDiscreteNavEKF(
        state0=state0,
        P0=P0_STD_COV,
        Q=Q_design,
        N=N_MEAS,
        dt=DT,
        landmarks=LANDMARKS,
        gravity=GRAVITY,
    )
    liekf = ContinuousDiscreteNavLIEKF(
        state0=state0,
        P0=P0_right,
        Q_base=Q_design,
        obs_cov=OBS_COV,
        dt=DT,
        landmarks=LANDMARKS,
        gravity=GRAVITY,
    )
    iscvx = ContinuousDiscreteNavISCVX(
        state0=state0,
        P0=P0_right,
        Q_base=Q_design,
        obs_cov=OBS_COV,
        dt=DT,
        landmarks=LANDMARKS,
        gravity=GRAVITY,
        trust_radius=0.75,
        max_scp_iters=5,
        solver="CLARABEL",
    )

    ekf_states = [ekf.state.copy()]
    liekf_states = [liekf.state.copy()]
    iscvx_states = [iscvx.state.copy()]

    for k in range(N_STEPS - 1):
        y_next = measurements[k + 1] if update_mask[k + 1] else None
        ekf_states.append(ekf.step(imu_inputs[k], y_next))
        liekf_states.append(liekf.step(imu_inputs[k], y_next))
        iscvx_states.append(iscvx.step(imu_inputs[k], y_next))

    def arrays(states):
        R = np.stack([s.R for s in states])
        v = np.stack([s.v for s in states])
        p = np.stack([s.p for s in states])
        return R, v, p

    true_R, true_v, true_p = arrays(true_states)
    ekf_R, ekf_v, ekf_p = arrays(ekf_states)
    liekf_R, liekf_v, liekf_p = arrays(liekf_states)
    iscvx_R, iscvx_v, iscvx_p = arrays(iscvx_states)

    ekf_att = np.array([attitude_error_deg(Rt, Rh) for Rt, Rh in zip(true_R, ekf_R)])
    liekf_att = np.array([attitude_error_deg(Rt, Rh) for Rt, Rh in zip(true_R, liekf_R)])
    iscvx_att = np.array([attitude_error_deg(Rt, Rh) for Rt, Rh in zip(true_R, iscvx_R)])
    ekf_pos = np.array([position_error(pt, ph) for pt, ph in zip(true_p, ekf_p)])
    liekf_pos = np.array([position_error(pt, ph) for pt, ph in zip(true_p, liekf_p)])
    iscvx_pos = np.array([position_error(pt, ph) for pt, ph in zip(true_p, iscvx_p)])

    return {
        "case_name": case_name,
        "time": time,
        "landmarks": LANDMARKS,
        "true_p": true_p,
        "ekf_p": ekf_p,
        "liekf_p": liekf_p,
        "iscvx_p": iscvx_p,
        "ekf_att_err_deg": ekf_att,
        "liekf_att_err_deg": liekf_att,
        "iscvx_att_err_deg": iscvx_att,
        "ekf_pos_err_m": ekf_pos,
        "liekf_pos_err_m": liekf_pos,
        "iscvx_pos_err_m": iscvx_pos,
        "iscvx_used_cvxpy": bool(iscvx.used_cvxpy),
        "ekf_att_rmse_deg": float(np.sqrt(np.mean(ekf_att**2))),
        "liekf_att_rmse_deg": float(np.sqrt(np.mean(liekf_att**2))),
        "iscvx_att_rmse_deg": float(np.sqrt(np.mean(iscvx_att**2))),
        "ekf_pos_rmse_m": float(np.sqrt(np.mean(ekf_pos**2))),
        "liekf_pos_rmse_m": float(np.sqrt(np.mean(liekf_pos**2))),
        "iscvx_pos_rmse_m": float(np.sqrt(np.mean(iscvx_pos**2))),
        "ekf_final_att_err_deg": float(ekf_att[-1]),
        "liekf_final_att_err_deg": float(liekf_att[-1]),
        "iscvx_final_att_err_deg": float(iscvx_att[-1]),
        "ekf_final_pos_err_m": float(ekf_pos[-1]),
        "liekf_final_pos_err_m": float(liekf_pos[-1]),
        "iscvx_final_pos_err_m": float(iscvx_pos[-1]),
    }


def Estimator_sim():
    return [run_case("Q1 tight", Q1), run_case("Q2 inflated", Q2)]


if __name__ == "__main__":
    results = Estimator_sim()
    print("Flat-earth navigation EKF / LIEKF / ISCVX-EKF comparison")
    print(f"  CVXPY available: {CVXPY_AVAILABLE}")
    print(f"  T = {T_FINAL:.1f} s, IMU = {IMU_FREQ} Hz, landmark observations = {OBS_FREQ} Hz")
    print(f"  landmarks = {LANDMARKS.tolist()}")
    print(f"  observation covariance block = {OBS_STD**2:.1e} I_3")
    print("  initial errors: attitude norm 15 deg, position norm 1 m, velocity exact")
    print()

    summary = []
    for res in results:
        metric_keys = [
            "ekf_att_rmse_deg", "liekf_att_rmse_deg", "iscvx_att_rmse_deg",
            "ekf_pos_rmse_m", "liekf_pos_rmse_m", "iscvx_pos_rmse_m",
            "ekf_final_att_err_deg", "liekf_final_att_err_deg", "iscvx_final_att_err_deg",
            "ekf_final_pos_err_m", "liekf_final_pos_err_m", "iscvx_final_pos_err_m",
        ]
        row = {"case_name": res["case_name"], "iscvx_used_cvxpy": res["iscvx_used_cvxpy"]}
        row.update({k: float(res[k]) for k in metric_keys})
        summary.append(row)
        print(res["case_name"])
        print(f"  EKF   attitude RMSE: {res['ekf_att_rmse_deg']:.4f} deg")
        print(f"  LIEKF attitude RMSE: {res['liekf_att_rmse_deg']:.4f} deg")
        print(f"  ISCVX attitude RMSE: {res['iscvx_att_rmse_deg']:.4f} deg")
        print(f"  EKF   position RMSE: {res['ekf_pos_rmse_m']:.4f} m")
        print(f"  LIEKF position RMSE: {res['liekf_pos_rmse_m']:.4f} m")
        print(f"  ISCVX position RMSE: {res['iscvx_pos_rmse_m']:.4f} m")
        print(f"  EKF   final errors: att={res['ekf_final_att_err_deg']:.4f} deg, pos={res['ekf_final_pos_err_m']:.4f} m")
        print(f"  LIEKF final errors: att={res['liekf_final_att_err_deg']:.4f} deg, pos={res['liekf_final_pos_err_m']:.4f} m")
        print(f"  ISCVX final errors: att={res['iscvx_final_att_err_deg']:.4f} deg, pos={res['iscvx_final_pos_err_m']:.4f} m")
        print(f"  ISCVX used CVXPY: {res['iscvx_used_cvxpy']}")
        print()

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_all_cases(results, save_path=str(out_dir / "flat_earth_all_cases_errors.png"))
    for res in results:
        name = res["case_name"].lower().replace(" ", "_")
        plot_case(res, save_path=str(out_dir / f"flat_earth_{name}_comparison.png"))
    plt.show()
