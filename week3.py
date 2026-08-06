import numpy as np
import math


def local_frame(E, I, A, L):
    beta = A * L * L / I
    return (
        np.array(
            [
                [beta, 0, 0, -beta, 0, 0],
                [0, 12, 6 * L, 0, -12, 6 * L],
                [0, 6 * L, 4 * L**2, 0, -6 * L, 2 * L**2],
                [-beta, 0, 0, beta, 0, 0],
                [0, -12, -6 * L, 0, 12, -6 * L],
                [0, 6 * L, 2 * L**2, 0, -6 * L, 4 * L**2],
            ]
        )
        * E
        * I
        / (L**3)
    )


def global_frame(K, alpha):
    c = math.cos(alpha)
    s = math.sin(alpha)

    inner = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    lambda_matrix = np.block(
        [[inner, np.zeros_like(inner)], [np.zeros_like(inner), inner]]
    )

    return lambda_matrix.T @ K @ lambda_matrix, lambda_matrix


E = 200e9
L = 10
I = 5e-4
A = 1e-5

K = local_frame(E, I, A, L)

K_hat_1, lambda_1 = global_frame(K, math.radians(-90))
K_hat_2, lambda_2 = global_frame(K, 0)

A_1 = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]
)

A_2 = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)


K_g1 = A_1 @ K_hat_1 @ A_1.T
K_g2 = A_2 @ K_hat_2 @ A_2.T


K_g = K_g1 + K_g2

Q = np.array([0, 140e3, 0])

q = np.linalg.solve(K_g, Q)

D_1 = A_1.T @ q
D_2 = A_2.T @ q

d_1 = lambda_1 @ D_1
d_2 = lambda_2 @ D_2

f_1 = K @ d_1
f_2 = K @ d_2

F_1 = K_hat_1 @ D_1
F_2 = K_hat_2 @ D_2

import matplotlib.pyplot as plt


def plot_deflected_shape(node1XG, node1YG, node2XG, node2YG, d_e, Disp_mag, N_points):
    x_e = np.linspace(0, L, N_points)

    phi_1 = 1 - x_e / L
    phi_2 = x_e / L

    N_1 = 1 - 3 * x_e**2 / L**2 + 2 * x_e**3 / L**3
    N_2 = x_e**3 / L**2 - 2 * x_e**2 / L + x_e
    N_3 = 3 * x_e**2 / L**2 - 2 * x_e**3 / L**3
    N_4 = x_e**3 / L**2 - x_e**2 / L

    u = phi_1 * d_e[0] + phi_2 * d_e[3]

    v = N_1 * d_e[1] + N_2 * d_e[2] + N_3 * d_e[4] + N_4 * d_e[5]

    alpha = math.atan2(node2YG - node1YG, node2XG - node1XG)

    Deflections_XG = u * math.cos(alpha) - v * math.sin(alpha)
    Deflections_YG = u * math.sin(alpha) + v * math.cos(alpha)

    Undeflected_baseline_XG = np.linspace(node1XG, node2XG, N_points)
    Undeflected_baseline_YG = np.linspace(node1YG, node2YG, N_points)

    Deflected_XG = Undeflected_baseline_XG + Disp_mag * Deflections_XG
    Deflected_YG = Undeflected_baseline_YG + Disp_mag * Deflections_YG

    plt.plot(Undeflected_baseline_XG, Undeflected_baseline_YG, "b.-")
    plt.plot(Deflected_XG, Deflected_YG, "r.-")


plot_deflected_shape(0, L, 0, 0, d_1, 100, 100)
plot_deflected_shape(0, 0, L, 0, d_2, 100, 100)
plt.show()
