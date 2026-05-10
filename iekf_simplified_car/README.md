# Simplified car EKF/LIEKF paper replication

This folder replicates the simplified-car example from Barrau and Bonnabel, *The Invariant Extended Kalman Filter as a Stable Observer*, TAC 2017, Section IV.

Run:

```bash
python main_script.py
```

The code uses the same setup stated in the paper's simulation section:

- unicycle dynamics
  \[
  \dot \theta = \omega,\qquad \dot x^{(1)}=\cos(\theta)v,\qquad \dot x^{(2)}=\sin(\theta)v,
  \]
- 10 m diameter circular trajectory for 40 s,
- odometry propagation at 100 Hz,
- GPS position observations at 1 Hz,
- \(N=I_2\),
- \(Q=\operatorname{diag}((\pi/180)^2,10^{-4},10^{-4})\),
- initial position known,
- two initial heading errors: \(1^\circ\) and \(45^\circ\).

Files:

- `dynamics.py`: unicycle dynamics, SE(2) maps, EKF/LIEKF linearization matrices.
- `integrator.py`: covariance integration helper.
- `continuous_discrete_ekf.py`: standard continuous-discrete EKF.
- `continuous_discrete_liekf.py`: left-invariant continuous-discrete EKF.
- `plotting.py`: figure helper.
- `main_script.py`: simulation and comparison driver.

The simulated process and GPS noises are off by default to reproduce the deterministic observer-style comparison; `Q` and `N` are still used as the filter design/tuning matrices, as in the paper.
