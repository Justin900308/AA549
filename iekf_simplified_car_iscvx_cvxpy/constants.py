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

# Paper parameters.
dt = 0.01  # 100 Hz odometry propagation
dt_traj_gen = 0.5
T_FINAL = 40.0  # seconds
T = int(T_FINAL / dt) + 1
T_traj_gen = int(T_FINAL / dt_traj_gen) + 1
GPS_DT = 4  # 1 Hz GPS
UPDATE_STRIDE = int(GPS_DT / dt)
n = 3
m = 2
num_obs = 2
obs = np.array([[4, 3], [8, 3]]) * 1
obs_r = 1.5
z_des = np.array([0, 10, 5.5])

CIRCLE_DIAMETER = 10.0  # meters
RADIUS = CIRCLE_DIAMETER / 2.0
omega_const = 1.5 * np.pi / T_FINAL
v_const = RADIUS * omega_const
Q = np.diag([(np.pi / 180.0) ** 2, 1e-4, 1e-4])  ## for process noise
N = np.eye(2)  ## for measurement noise
H0 = np.array([np.pi / 2.0, RADIUS, 0.0])  # starts on a radius-5 circle centered at origin

# The paper's Fig. 1 uses 1 degree and 45 degree initial heading errors.
INITIAL_HEADING_ERRORS_DEG = [1.0, 45.0]

# Keep actual process and measurement noise off to reproduce the deterministic observer comparison.
# Q and N are still used as EKF/LIEKF design/tuning matrices, as in the paper.
ADD_SIMULATION_NOISE = False  ## Turn on or off the process noise
RNG_SEED = 13
