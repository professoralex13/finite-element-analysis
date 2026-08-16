import numpy as np
import math

L_1 = 10
L_2 = 14.1
E = 200e9


def local_bar(E, A, L):
    return (
        np.array(
            [
                [1, -1],
                [-1, 1],
            ]
        )
        * E
        * A
        / L
    )


def global_bar(K, alpha):
    s = math.sin(alpha)
    c = math.cos(alpha)

    lambda_m = np.array([[c, s, 0, 0], [0, 0, c, s]])

    return lambda_m.T @ K @ lambda_m, lambda_m


K_1 = local_bar(E, math.pi * 0.05**2, L_1)
K_2 = local_bar(E, math.pi * 0.05**2, L_2)

K_hat_1, lambda_1 = global_bar(K_1, 0)
K_hat_2, lambda_2 = global_bar(K_2, math.pi / 4)

A_1 = np.matrix([[0, 0, 1, 0], [0, 0, 0, 1]])

A_2 = np.matrix([[0, 0, 1, 0], [0, 0, 0, 1]])

K_global_1 = A_1 @ K_hat_1 @ A_1.T
K_global_2 = A_2 @ K_hat_2 @ A_2.T

K_global = K_global_1 + K_global_2

Q = np.matrix([0, 1e5]).T

q = np.linalg.solve(K_global, Q)

F_1 = K_hat_1 @ A_1.T @ q
F_2 = K_hat_2 @ A_2.T @ q

d_1 = lambda_1 @ A_1.T @ q
d_2 = lambda_2 @ A_2.T @ q

strain_1 = (d_1[1, 0] - d_1[0, 0]) / L_1
strain_2 = (d_2[1, 0] - d_2[0, 0]) / L_2

f_1 = K_1 @ d_1
f_2 = K_2 @ d_2

print(f_2)
