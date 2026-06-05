"""Continuous-discrete particle filter for the simplified car.

Drop-in alternative to ``ContinuousDiscreteCarEKF``.

State convention is the same as the EKF code:
    z = [theta, x, y]
Measurement model is GPS position:
    y_gps = [x, y] + noise,  noise ~ N(0, N)

The algorithm follows the particle-filter structure in the reference code:
    1. resample particles using current weights,
    2. propagate each particle through the nonlinear dynamics plus process noise,
    3. weight predicted particles by the measurement likelihood,
    4. estimate state by the weighted particle mean.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from dynamics import wrap_angle, unicycle_dynamics
from integrator import rk4


class ContinuousDiscreteCarParticleFilter:
    def __init__(
        self,
        z0,
        P0,
        Q,
        N,
        dt: float,
        num_particles: int = 1000,
        rng: Optional[np.random.Generator] = None,
        resample_threshold: float = 0.5,
        process_noise_discrete: bool = False,
    ) -> None:
        self.z = np.asarray(z0, dtype=float).reshape(3).copy()
        self.z[0] = wrap_angle(self.z[0])

        self.P = np.asarray(P0, dtype=float).reshape(3, 3).copy()
        self.Q = np.asarray(Q, dtype=float).reshape(3, 3).copy()
        self.N = np.asarray(N, dtype=float).reshape(2, 2).copy()
        self.dt = float(dt)

        self.num_particles = int(num_particles)

        self.rng = np.random.default_rng() if rng is None else rng
        self.resample_threshold = float(resample_threshold)
        self.process_noise_discrete = bool(process_noise_discrete)

        self.weights = np.ones(self.num_particles, dtype=float) / self.num_particles
        self.particles = self.rng.multivariate_normal(self.z, self.P, size=self.num_particles)
        self.particles[:, 0] = np.vectorize(wrap_angle)(self.particles[:, 0])

        self.N_inv = np.linalg.inv(self.N)
        sign, logdet = np.linalg.slogdet(self.N)
        if sign <= 0:
            raise ValueError("measurement covariance N must be positive definite")
        self._meas_log_norm = -0.5 * (2 * np.log(2.0 * np.pi) + logdet)

        self.update_t = None

    def _measurement_model(self, particles: np.ndarray) -> np.ndarray:
        return particles[:, 1:3]

    def _estimate_from_particles(self) -> None:
        """Update self.z and self.P from the weighted particle cloud."""
        w = self.weights
        theta_mean = np.arctan2(
            np.sum(w * np.sin(self.particles[:, 0])),
            np.sum(w * np.cos(self.particles[:, 0])),
        )
        xy_mean = w @ self.particles[:, 1:3]
        self.z = np.array([theta_mean, xy_mean[0], xy_mean[1]])
        self.z[0] = wrap_angle(self.z[0])

        err = self.particles - self.z
        err[:, 0] = np.vectorize(wrap_angle)(err[:, 0])
        self.P = (err.T * w) @ err
        self.P = 0.5 * (self.P + self.P.T) + 1e-15 * np.eye(3)

    def effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights**2)

    def systematic_resample(self) -> None:
        """Low-variance/systematic resampling."""
        n = self.num_particles
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # avoid roundoff edge case
        indexes = np.searchsorted(cumulative_sum, positions)
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / n)

    def predict(self, u) -> None:
        u = np.asarray(u, dtype=float).reshape(2)

        propagated = np.empty_like(self.particles)
        ## propagate the particles
        for i in range(self.num_particles):
            propagated[i] = rk4(unicycle_dynamics, self.particles[i], u, self.dt)

        Qd = self.Q if self.process_noise_discrete else self.Q * self.dt
        noise = self.rng.multivariate_normal(np.zeros(3), Qd, size=self.num_particles)
        self.particles = propagated + noise
        self.particles[:, 0] = np.vectorize(wrap_angle)(self.particles[:, 0])

        self._estimate_from_particles()

    def update(self, y_gps) -> None:
        t0 = time.time()
        y_gps = np.asarray(y_gps, dtype=float).reshape(2)

        self.systematic_resample()

        residual = y_gps[None, :] - self._measurement_model(self.particles)

        ## compute log likelyhood
        quad = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            r = residual[i]
            quad[i] = r.T @ self.N_inv @ r
        log_likelihood = self._meas_log_norm - 0.5 * quad

        log_weights = np.log(self.weights + 1e-20) + log_likelihood
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weight_sum = np.sum(weights)

        if (not np.isfinite(weight_sum)) or weight_sum <= 0.0:
            # Degenerate likelihood; recover safely instead of producing NaNs.
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights = weights / weight_sum

        self._estimate_from_particles()
        self.update_t = time.time() - t0

    def step(self, u, y_gps=None) -> np.ndarray:
        self.predict(u)
        if y_gps is not None:
            self.update(y_gps)
        return self.z.copy()
