# Continuous-discrete quaternion EKF and LIEKF

Run the comparison with:

```bash
python main_script.py
```

The conditioning step is performed every `UPDATE_STRIDE * dt = 10 * dt`.

Files:

- `dynamics.py`: quaternion operations, dynamics, vector measurement model.
- `integrator.py`: RK4 step for continuous quaternion propagation.
- `linearization_lib.py`: finite-difference EKF Jacobian and intrinsic LIEKF Jacobian.
- `continuous_discrete_ekf.py`: additive continuous-discrete quaternion EKF.
- `continuous_discrete_liekf.py`: left-invariant continuous-discrete quaternion EKF.
- `plotting.py`: comparison plotting and optional 3D attitude animation.
- `main_script.py`: simulation and EKF/LIEKF comparison.
