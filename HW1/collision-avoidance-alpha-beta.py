import numpy as np
import matplotlib.pyplot as plt

dt = 0.1  # sampling time
T = 100  # total time steps

# filter gain parameters
para = []
para.append((0.1, 0.2))
para.append((0.5, 1.0))
para.append((1.0, 2.0))


def simulation(case):
    alpha = para[case][0]
    beta = para[case][1]

    # model parameters
    F = np.array([[1, dt],
                  [0, 1]])
    G = np.array([dt ** 2 / 2, dt])
    H = np.array([1, 0])
    K = np.array([[alpha], [beta / dt]])

    X = np.zeros([T, 2])  # true state
    X_hat = np.zeros([T, 2])  # estimate
    Y = np.zeros([T, 1])  # measurement

    X[0, :] = [1, 10]  # initial condition

    for i in range(T - 1):
        d = np.random.randn()  # process noise
        X[i + 1] = F @ X[i] + G * d  # dynamic model
        W = 0.01 * np.random.randn()  # measurement noise
        Y[i + 1] = H @ X[i + 1] + W  # measurement model

        # estiamtion algorithm
        X_hat[i + 1] = F @ X_hat[i]  # prediction
        Y_hat = H @ X_hat[i + 1]
        X_hat[i + 1] = X_hat[i + 1] + K @ (Y[i + 1] - Y_hat)  # correction

    return X, Y, X_hat


def plotting(X, Y, X_hat, case):
    alpha = para[case][0]
    beta = para[case][1]
    plt.subplot(2, 1, 1)
    plt.plot(X[:, 0], label='distance')
    plt.plot(Y, label='measurement', ls='None', marker='x', ms=5)
    plt.plot(X_hat[:, 0], label='estimate')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(X[:, 1], label='relative speed')
    plt.plot(X_hat[:, 1], label='estimate', color='C2')
    plt.legend()
    plt.savefig('alpha-%.1f-beta-%.1f.png' % (alpha, beta))
    plt.show()


for i in range(3):
    case = i
    X, Y, X_hat = simulation(case)
    plotting(X, Y, X_hat, case)
