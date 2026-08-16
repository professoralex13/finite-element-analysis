"""
FEA Truss solver: Alex Cutforth July 2026
"""

import math
import numpy as np


def solve_truss(node_positions, dofs, elements):
    """Solves a truss system using FEA"""

    dof_keys = list(dofs.keys())

    dof_forces = np.array(list(dofs.values()))

    n_dof = len(dof_forces)

    local_stiffness_matrices = []
    global_oriented_stiffness_matrices = []
    transfomation_matrices = []
    assembly_matrices = []

    global_nodal_stiffness_matrices = []

    for start, end, stiffness in elements:
        vector = node_positions[end] - node_positions[start]
        alpha = math.atan2(vector[1], vector[0])

        local_stiffness_matrix = (
            np.array(
                [
                    [1, -1],
                    [-1, 1],
                ]
            )
            * stiffness
            / np.linalg.norm(vector)
        )

        local_stiffness_matrices.append(local_stiffness_matrix)

        s = math.sin(alpha)
        c = math.cos(alpha)

        _lambda = np.array([[c, s, 0, 0], [0, 0, c, s]])

        global_nodal_stiffness_matrices = _lambda.T @ local_stiffness_matrix @ _lambda

        global_oriented_stiffness_matrices.append(global_nodal_stiffness_matrices)
        transfomation_matrices.append(_lambda)

        assembly_matrix = np.zeros((n_dof, 4))

        if f"{start}/x" in dofs:
            assembly_matrix[dof_keys.index(f"{start}/x"), 0] = 1

        if f"{start}/y" in dofs:
            assembly_matrix[dof_keys.index(f"{start}/y"), 1] = 1

        if f"{end}/x" in dofs:
            assembly_matrix[dof_keys.index(f"{end}/x"), 2] = 1

        if f"{end}/y" in dofs:
            assembly_matrix[dof_keys.index(f"{end}/y"), 3] = 1

        assembly_matrices.append(assembly_matrix)

        global_nodal_stiffness_matrices.append(
            assembly_matrix @ global_nodal_stiffness_matrices @ assembly_matrix.T
        )

    total_stiffness_matrix = sum(global_nodal_stiffness_matrices)

    dof_deflections = np.linalg.solve(total_stiffness_matrix, dof_forces)

    local_force_vectors = [
        K_hat @ A.T @ dof_deflections
        for K_hat, A in zip(global_oriented_stiffness_matrices, assembly_matrices)
    ]

    local_deflection_vectors = [
        _lambda @ A.T @ dof_deflections
        for _lambda, A in zip(transfomation_matrices, assembly_matrices)
    ]

    return dof_deflections


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
