"""
FEA Truss solver: Alex Cutforth July 2026
"""

import math
import numpy as np


def solve_truss(node_positions, dofs, elements):
    dof_keys = list(dofs.keys())

    Q = np.array(list(dofs.values()))

    n_dof = len(Q)

    K_n = []
    K_hat_n = []
    lambda_n = []
    A_n = []

    K_global_n = []

    for start, end, stiffness in elements:
        vector = node_positions[end] - node_positions[start]
        alpha = math.atan2(vector[1], vector[0])

        K = (
            np.array(
                [
                    [1, -1],
                    [-1, 1],
                ]
            )
            * stiffness
            / np.linalg.norm(vector)
        )

        K_n.append(K)

        s = math.sin(alpha)
        c = math.cos(alpha)

        _lambda = np.array([[c, s, 0, 0], [0, 0, c, s]])

        K_hat = _lambda.T @ K @ _lambda

        K_hat_n.append(K_hat)
        lambda_n.append(_lambda)

        A = np.zeros((n_dof, 4))

        if f"{start}/x" in dofs:
            A[dof_keys.index(f"{start}/x"), 0] = 1

        if f"{start}/y" in dofs:
            A[dof_keys.index(f"{start}/y"), 1] = 1

        if f"{end}/x" in dofs:
            A[dof_keys.index(f"{end}/x"), 2] = 1

        if f"{end}/y" in dofs:
            A[dof_keys.index(f"{end}/y"), 3] = 1

        A_n.append(A)

        K_global_n.append(A @ K_hat @ A.T)

    K_global = sum(K_global_n)

    q = np.linalg.solve(K_global, Q)

    f_n = [K_hat @ A.T @ q for K_hat, A in zip(K_hat_n, A_n)]

    d_n = [_lambda @ A.T @ q for _lambda, A in zip(lambda_n, A_n)]

    return q


E = 200e9
A1 = 400e-6
A2 = 600e-6

L1 = 1.1
L2 = 0.8


q = solve_truss(
    {
        "A": np.array([-L1, 0]),
        "B": np.array([0, 0]),
        "C": L2
        * np.array([math.cos(55 * math.pi / 180), math.sin(55 * math.pi / 180)]),
    },
    {"B/x": 0, "B/y": -20e3},
    [("A", "B", E * A1), ("B", "C", E * A2)],
)

print(q)
