import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from utilities import dynamics as dy
import numpy.linalg as LA


def NED_2_plot_frame(v):
    return np.array([v[1], v[0], -v[2]])


def plotting3d(q_traj, omega_traj, T):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    for t in range(T):
        qt = q_traj[t]

        ## plotting the attitude
        x_body_t = NED_2_plot_frame(dy.quat_rotate(qt, x_axis))
        x_body_t = x_body_t / LA.norm(x_body_t, 2)
        y_body_t = NED_2_plot_frame(dy.quat_rotate(qt, y_axis))
        y_body_t = y_body_t / LA.norm(y_body_t, 2)
        z_body_t = NED_2_plot_frame(dy.quat_rotate(qt, z_axis))
        z_body_t = z_body_t / LA.norm(z_body_t, 2)
        ax.plot(np.array([0, x_body_t[0]]),
                np.array([0, x_body_t[1]]),
                + np.array([0, x_body_t[2]]), "r")
        ax.plot(np.array([0, y_body_t[0]]),
                np.array([0, y_body_t[1]]),
                np.array([0, y_body_t[2]]), "g")
        ax.plot(np.array([0, z_body_t[0]]),
                np.array([0, z_body_t[1]]),
                np.array([0, z_body_t[2]]), "b")

        if t < T - 1:
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            ax.set_xlabel("East")
            ax.set_ylabel("North")
            ax.set_zlabel("Up")
            plt.pause(0.01)
            ax.clear()

        else:
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            ax.set_xlabel("East")
            ax.set_ylabel("North")
            ax.set_zlabel("Up")
            plt.show()
