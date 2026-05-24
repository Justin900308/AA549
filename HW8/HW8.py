import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import csv
import scipy as sp

#### constants
nx = 3
nu = 2
ny = 2
alpha = 0.03
beta = 0.5
Q = alpha ** 2 * np.eye(nx)
R = beta ** 2 * np.eye(ny)
T = 200
steps = np.linspace(0, T - 1, T)
u = np.array([0.1, 0.02])
## generate random numbers
mu = 0
sigma = 1
R = np.eye(ny)
np.random.seed(225678)

#### load the data
X_data = np.zeros((T, nx))
Y_data = np.zeros((T, ny))
with open('HW6_combined.csv', mode='r') as file:
    csvFile = csv.reader(file)
    i = 0
    for lines in csvFile:
        X_data[i] = np.array([float(lines[0]), float(lines[1]), float(lines[2])])
        Y_data[i] = np.array([float(lines[3]), float(lines[4])])
        i += 1
## generate the initial particles
Num_particles = [100, 100, 1000]
particle_dist = [[np.array([2, 0, np.pi / 2]), np.eye(nx)],
                 [np.array([0, 0, 0]), np.eye(nx)],
                 [np.array([0, 0, 0]), np.eye(nx)]]
particles = [np.zeros([T, Num_particles[0], nx]),
             np.zeros([T, Num_particles[1], nx]),
             np.zeros([T, Num_particles[2], nx])]

## initial particle dist
for i in range(len(Num_particles)):
    mean_i = particle_dist[i][0]
    cov_i = particle_dist[i][1]
    N = Num_particles[i]
    particles[i][0] = np.random.multivariate_normal(mean_i, cov_i, size=N)


## the observation model
def h_fun(Xi):
    hi = np.array([np.sqrt((Xi[0] - 1) ** 2 + Xi[1] ** 2),
                   np.sqrt((Xi[0] + 1) ** 2 + Xi[1] ** 2)])
    return hi


def gamma_fun(y, Xi):
    coef = 1 / np.sqrt((2 * np.pi) ** ny * np.linalg.det(R))
    hi = h_fun(Xi)
    alpha = -0.5 * (y - hi).T @ np.linalg.inv(R) @ (y - hi)
    gamma = coef * np.exp(alpha)
    return gamma


def compute_weight(wt, particles_t, yt):
    ## conditioning
    wtp1 = np.zeros(len(particles_t))
    for i in range(len(particles_t)):
        wtp1[i] = wt[i] * gamma_fun(yt, particles_t[i])
    ## normalization
    wtp1 = wtp1 / np.sum(wtp1)
    return wtp1


def redraw_sample(N, w, particles_t):
    idx = np.arange(N)
    idx_gen = np.random.choice(idx, p=w, replace=True, size=N)
    particles_gen = particles_t[idx_gen]
    return particles_gen


def prediction(N, particles_t):
    v = np.random.normal(mu, sigma, (N, nx))
    ## update particles
    particles_tp1 = np.zeros([N, nx])
    for i in range(N):
        Xt = np.array([particles_t[i, 0], particles_t[i, 1], particles_t[i, 2]])
        particles_tp1[i] = np.array([Xt[0] + u[0] * np.cos(Xt[2]) + alpha * v[i, 0],
                                     Xt[1] + u[0] * np.sin(Xt[2]) + alpha * v[i, 1],
                                     Xt[2] + u[1] + alpha * v[i, 2]])
    return particles_tp1


def particle_filter(particles, Y_data):
    N = len(particles[0, :, 0])
    w = np.ones(N) / N  ## Initial weight
    X_est = np.zeros([T, nx])
    ## Get initial estimation
    X_est[0, :] = w @ particles[0]
    for t in range(T - 1):
        ## regenerate particle
        particles[t] = redraw_sample(N, w, particles[t])
        ## reset the current weight
        w = np.ones(N) / N
        ## compute the next weight
        particles[t + 1] = prediction(N, particles[t])  ## particle pred
        w = compute_weight(w, particles[t + 1], Y_data[t])  ## compute weight from pred
        ## Get estimation
        X_est[t + 1] = w @ particles[t + 1]
    return X_est


## compute position mse
def MSE_compute(X_est_i, X_data):
    MSE_i = (X_est_i[:, 0] - X_data[:, 0]) ** 2 + (X_est_i[:, 1] - X_data[:, 1]) ** 2
    return MSE_i


Num_cases = 3
X_est = []
MSE = []
for i in range(Num_cases):
    X_est.append(particle_filter(particles[i], Y_data))
    MSE.append(MSE_compute(X_est[i], X_data))

#### plotting
fig, ax = plt.subplots(Num_cases, 2, figsize=(8, 12))
for i in range(Num_cases):
    ## for position
    ax[i, 0].plot(X_data[:, 0], X_data[:, 1], "r.", markersize=10, label="True traj")
    ax[i, 0].plot(X_est[i][:, 0], X_est[i][:, 1], "b.", markersize=10, label=rf"Estimates case {i + 1}")
    ax[i, 0].set_xlabel("x (m)")
    ax[i, 0].set_ylabel("y (m)")
    ax[i, 0].legend()
    ## for heading
    ax[i, 1].plot(steps, X_data[:, 2] * 180 / np.pi, "r.", markersize=10, label="True traj")
    ax[i, 1].plot(steps, X_est[i][:, 2] * 180 / np.pi, "b.", markersize=10, label=rf"Estimates case {i + 1}")
    ax[i, 1].set_xlabel("step (k)")
    ax[i, 1].set_ylabel("heading (deg)")
    ax[i, 1].legend()

plt.tight_layout()
plt.savefig('state_compare.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(4, 4))
for i in range(Num_cases):
    ax.plot(steps, MSE[i], ".", markersize=10, label=rf"Position MSE case {i + 1}")
    ax.legend()
    ax.set_xlabel("step (k)")
    ax.set_ylabel("MSE")
plt.tight_layout()
plt.savefig('MSE_compare.pdf')
plt.show()
