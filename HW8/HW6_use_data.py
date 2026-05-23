import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import csv

#### constants
nx = 3
nu = 2
ny = 2
alpha = 0.03
beta = 0.5
Q = alpha ** 2 * np.eye(nx)
R = beta ** 2 * np.eye(ny)
T = 200
u = np.array([0.1, 0.02])
## generate random numbers
mu = 0
sigma = 1
v = np.random.normal(mu, sigma, (T, nx))
w = np.random.normal(mu, sigma, (T - 1, ny))
# np.random.seed(12345678)


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
    mu_particle = particles[i][0]
    N = Num_particles[i]
    particles[i][0] = mu_particle + np.random.normal(0, 1, (N, nx))


def compute_weight():
    return


def redraw_sample(N, w, particles_t):
    idx = np.arange(N)
    idx_gen = np.random.choice(idx, p=w, replace=True, size=N)
    particles_gen = particles_t[idx_gen]
    return particles_gen


def prediction(N, particles):
    v = np.random.normal(mu, sigma, (N, nx))
    ## update particles
    for i in range(N):
        Xt = np.array([particles[i][0], particles[i][1], particles[i][2]])
        particles[i] = np.array([Xt[0] + u[0] * np.cos(Xt[2]) + alpha * v[i, 0],
                                 Xt[1] + u[0] * np.sin(Xt[2]) + alpha * v[i, 1],
                                 Xt[2] + u[1] + alpha * v[i, 2]])
    return particles


def particle_filter(particles, X_data, Y_data):
    N = len(particles[0, :, 0])
    w = np.ones(N) / N  ## Initial weight
    X_est = np.zeros([T, nx])
    ## Get initial estimation
    X_est[0, :] = np.sum(w @ particles[0])
    for t in range(T - 1):
        ## regenerate particle
        particles[t] = redraw_sample(N, w, particles[t])
        ## reset the current weight
        w = np.ones(N) / N  ## Initial weight
        ## compute the next weight
        particles[t + 1] = prediction(N, particles[t])  ## particle pred
        w = compute_weight(w, particles[t + 1])  ## compute weight from pred
        ## Get estimation
        X_est[t + 1, :] = np.sum(w @ particles[t + 1])
    return X_est


X_est = particle_filter(particles[0], X_data, Y_data)
