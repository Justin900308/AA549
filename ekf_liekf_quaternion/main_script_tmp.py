"""Run and compare the continuous-discrete quaternion EKF and LIEKF.

The conditioning step is performed every UPDATE_STRIDE * dt = 10 * dt.
Run:
    python main_script.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from continuous_discrete_ekf import ContinuousDiscreteQuaternionEKF
from continuous_discrete_liekf import ContinuousDiscreteQuaternionLIEKF
from dynamics import (
    attitude_error_angle,
    measurement_model_left,
    quat_dynamics,
    quat_exp,
    quat_mul,
    normalize_quat,
)
from integrator import RK4
from plotting import plot_filter_comparison


# Simulation constants.
dt = 0.02
T = 1000
nx = 4
nu = 3
UPDATE_STRIDE = 10                 # conditioning/update period = 10 dt
UPDATE_DT = UPDATE_STRIDE * dt

# Two non-collinear body-frame reference vectors.  A single vector leaves one
# rotation direction unobservable, so two vectors make the comparison meaningful.
REFERENCE_BODY_VECTORS = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=float,
)
ny = 3 * len(REFERENCE_BODY_VECTORS)

# Noise levels.  Units: gyro noise is rad/s; vector measurement noise is unit-vector noise.
gyro_sigma = 0.015
meas_sigma = 0.035
Q_gyro = gyro_sigma ** 2 * np.eye(3)
R_meas = meas_sigma ** 2 * np.eye(ny)


rng = np.random.default_rng(7)


def omega_profile(t: float) -> np.ndarray:
    """Body angular velocity used to generate the true trajectory."""
    return np.array(
        [
            0.18 * np.sin(0.55 * t) + 0.03,
            0.06 + 0.12 * np.cos(0.21 * t),
            0.11 * np.sin(0.33 * t + 0.4),
        ],
        dtype=float,
    )


def traj_simulation():
    """Simulate true q, measured gyro input, and low-rate vector observations."""
    q_traj = np.zeros((T, nx))
    q_traj[0] = np.array([1.0, 0.0, 0.0, 0.0])

    omega_true = np.zeros((T - 1, nu))
    omega_meas = np.zeros((T - 1, nu))
    y_meas = np.full((T, ny), np.nan)
    update_mask = np.zeros(T, dtype=bool)

    for k in range(T - 1):
        tk = k * dt
        omega_true[k] = omega_profile(tk)
        omega_meas[k] = omega_true[k] + rng.normal(0.0, gyro_sigma, 3)
        q_traj[k + 1] = RK4(q_traj[k], omega_true[k], quat_dynamics, dt)

        if (k + 1) % UPDATE_STRIDE == 0:
            y_clean = measurement_model_left(q_traj[k + 1], REFERENCE_BODY_VECTORS)
            y_meas[k + 1] = y_clean + rng.normal(0.0, meas_sigma, ny)
            update_mask[k + 1] = True

    return q_traj, omega_true, omega_meas, y_meas, update_mask


def Estimator_sim():
    """Run both estimators on the same measurements."""
    q_true, omega_true, omega_meas, y_meas, update_mask = traj_simulation()

    # A deliberately non-small initial error makes the geometric update difference visible.
    initial_error = np.deg2rad(np.array([70.0, -45.0, 55.0]))
    q0_est = normalize_quat(quat_mul(q_true[0], quat_exp(initial_error)))

    P0_ekf = 0.20 ** 2 * np.eye(4)
    P0_liekf = np.deg2rad(30.0) ** 2 * np.eye(3)

    ekf = ContinuousDiscreteQuaternionEKF(
        q0=q0_est,
        P0=P0_ekf,
        Q_gyro=Q_gyro,
        R_meas=R_meas,
        reference_body_vectors=REFERENCE_BODY_VECTORS,
        dt=dt,
    )
    liekf = ContinuousDiscreteQuaternionLIEKF(
        q0=q0_est,
        P0=P0_liekf,
        Q_gyro=Q_gyro,
        R_meas=R_meas,
        reference_body_vectors=REFERENCE_BODY_VECTORS,
        dt=dt,
    )

    q_ekf = np.zeros_like(q_true)
    q_liekf = np.zeros_like(q_true)
    q_ekf[0] = ekf.q
    q_liekf[0] = liekf.q

    for k in range(T - 1):
        yk = y_meas[k + 1] if update_mask[k + 1] else None
        q_ekf[k + 1] = ekf.step(omega_meas[k], yk)
        q_liekf[k + 1] = liekf.step(omega_meas[k], yk)

    time = np.arange(T) * dt
    ekf_err = np.array([attitude_error_angle(qt, qe) for qt, qe in zip(q_true, q_ekf)])
    liekf_err = np.array([attitude_error_angle(qt, ql) for qt, ql in zip(q_true, q_liekf)])

    results = {
        "time": time,
        "q_true": q_true,
        "omega_true": omega_true,
        "omega_meas": omega_meas,
        "y_meas": y_meas,
        "update_mask": update_mask,
        "q_ekf": q_ekf,
        "q_liekf": q_liekf,
        "ekf_error_deg": np.rad2deg(ekf_err),
        "liekf_error_deg": np.rad2deg(liekf_err),
        "ekf_rmse_deg": float(np.sqrt(np.mean(np.rad2deg(ekf_err) ** 2))),
        "liekf_rmse_deg": float(np.sqrt(np.mean(np.rad2deg(liekf_err) ** 2))),
        "ekf_final_error_deg": float(np.rad2deg(ekf_err[-1])),
        "liekf_final_error_deg": float(np.rad2deg(liekf_err[-1])),
        "update_dt": UPDATE_DT,
    }
    return results


if __name__ == "__main__":
    results = Estimator_sim()
    print(f"conditioning/update period: {UPDATE_STRIDE} dt = {results['update_dt']:.3f} s")
    print(f"additive EKF attitude RMSE: {results['ekf_rmse_deg']:.3f} deg")
    print(f"LIEKF attitude RMSE:       {results['liekf_rmse_deg']:.3f} deg")
    print(f"additive EKF final error:  {results['ekf_final_error_deg']:.3f} deg")
    print(f"LIEKF final error:         {results['liekf_final_error_deg']:.3f} deg")

    plot_filter_comparison(
        results["time"],
        results["q_true"],
        results["q_ekf"],
        results["q_liekf"],
        update_mask=results["update_mask"],
        save_path="ekf_liekf_comparison.png",
    )
    plt.show()
