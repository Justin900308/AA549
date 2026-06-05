# Simplified car EKF/LIEKF/ISCVX comparison with a CVXPY ISCVX backend

This folder is the CVXPY-oriented version of the simplified-car example from
Barrau and Bonnabel, *The Invariant Extended Kalman Filter as a Stable Observer*,
Section IV.

Run:

```bash
python main_script.py
```

The three compared filters are:

1. standard continuous-discrete EKF,
2. continuous-discrete LIEKF,
3. continuous-discrete ISCVX filter whose prediction is the same as the LIEKF
   and whose GPS conditioning step is solved by intrinsic successive
   convexification.

The ISCVX update solves the convexified subproblem

\[
\min_\eta
\|r_p + J_p\eta\|_{P^{-1}}^2
+
\|r_y - H\eta\|_{N^{-1}}^2,
\qquad
\|\eta\|_2 \le r,
\]

using CVXPY when CVXPY is installed.  The implementation is in
`continuous_discrete_iscvx_cvxpy.py`.  For portability, the class has
`fallback_without_cvxpy=True` by default, which uses the direct trust-region
solver if CVXPY is absent.  To force the actual CVXPY backend, instantiate

```python
ContinuousDiscreteCarISCVXCVXPY(..., fallback_without_cvxpy=False)
```

and install the dependency:

```bash
pip install cvxpy
```

The paper-matching simulation parameters are unchanged:

- unicycle dynamics,
- 10 m diameter circular trajectory for 40 s,
- odometry propagation at 100 Hz,
- GPS position observations at 1 Hz,
- \(N=I_2\),
- \(Q=\operatorname{diag}((\pi/180)^2,10^{-4},10^{-4})\),
- initial position known,
- initial heading errors \(1^\circ\) and \(45^\circ\).

Files:

- `continuous_discrete_iscvx_cvxpy.py`: CVXPY-backed intrinsic SCP update.
- `continuous_discrete_iscvx.py`: dependency-free direct trust-region version.
- `continuous_discrete_liekf.py`: paper-style LIEKF.
- `continuous_discrete_ekf.py`: standard EKF.
- `dynamics.py`: unicycle, SE(2), retraction/log, and Jacobian utilities.
- `plotting.py`: comparison plots.
- `main_script.py`: comparison driver.
