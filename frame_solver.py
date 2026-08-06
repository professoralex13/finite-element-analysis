import numpy as np
import math


class Node:
    def __init__(
        self,
        name: str,
        position,
        x: float | None,
        y: float | None,
        moment: float | None,
    ):
        self.name = name
        self.position = position
        self.x = x
        self.y = y
        self.moment = moment


class FrameElement:
    start: Node
    end: Node
    area: float
    second_moment_of_inertia: float
    elastic_modulus: float

    def __init__(
        self,
        start: Node,
        end: Node,
        area: float,
        second_moment_of_inertia: float,
        elastic_modulus: float,
    ):
        self.start = start
        self.end = end
        self.area = area
        self.second_moment_of_inertia = second_moment_of_inertia
        self.elastic_modulus = elastic_modulus


def solve_frame(elements: list[FrameElement]):
    nodes = list(dict.fromkeys([x for e in elements for x in (e.start, e.end)]))

    Q = np.array([])

    x_indices = {}
    y_indices = {}
    moment_indices = {}

    for node in nodes:
        if node.x is not None:
            Q = np.append(Q, node.x)
            x_indices[node.name] = len(Q) - 1

        if node.y is not None:
            Q = np.append(Q, node.y)
            y_indices[node.name] = len(Q) - 1

        if node.moment is not None:
            Q = np.append(Q, node.moment)
            moment_indices[node.name] = len(Q) - 1

    n_dof = len(Q)

    K_n = []
    K_hat_n = []
    lambda_n = []
    A_n = []

    K_global_n = []

    for element in elements:
        start = element.start
        end = element.end

        vector = end.position - start.position
        alpha = math.atan2(vector[1], vector[0])

        L = np.linalg.norm(vector)

        beta = element.area * L**2 / element.second_moment_of_inertia

        coef = element.elastic_modulus * element.second_moment_of_inertia / (L**3)

        K = coef * np.array(
            [
                [beta, 0, 0, -beta, 0, 0],
                [0, 12, 6 * L, 0, -12, 6 * L],
                [0, 6 * L, 4 * L**2, 0, -6 * L, 2 * L**2],
                [-beta, 0, 0, beta, 0, 0],
                [0, -12, -6 * L, 0, 12, -6 * L],
                [0, 6 * L, 2 * L**2, 0, -6 * L, 4 * L**2],
            ]
        )

        K_n.append(K)

        s = math.sin(alpha)
        c = math.cos(alpha)

        inner = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        lambda_matrix = np.block(
            [[inner, np.zeros_like(inner)], [np.zeros_like(inner), inner]]
        )

        K_hat = lambda_matrix.T @ K @ lambda_matrix

        K_hat_n.append(K_hat)
        lambda_n.append(lambda_matrix)

        A = np.zeros((n_dof, 6))

        if start.name in x_indices:
            A[x_indices[start.name], 0] = 1

        if start.name in y_indices:
            A[y_indices[start.name], 1] = 1

        if start.name in moment_indices:
            A[moment_indices[start.name], 2] = 1

        if end.name in x_indices:
            A[x_indices[end.name], 3] = 1

        if end.name in y_indices:
            A[y_indices[end.name], 4] = 1

        if end.name in moment_indices:
            A[moment_indices[end.name], 5] = 1

        A_n.append(A)

        K_global_n.append(A @ K_hat @ A.T)

    K_global = sum(K_global_n)

    q = np.linalg.solve(K_global, Q)

    f_n = [K_hat @ A.T @ q for K_hat, A in zip(K_hat_n, A_n)]

    d_n = [_lambda @ A.T @ q for _lambda, A in zip(lambda_n, A_n)]

    return q


L = 10
A = 1e-5
I = 5e-4
E = 200e9

node_a = Node("A", np.array([0, L]), None, None, None)
node_b = Node("B", np.array([0, 0]), 0, None, 140e3)
node_c = Node("C", np.array([L, 0]), None, None, 0)

element_1 = FrameElement(node_a, node_b, A, I, E)
element_2 = FrameElement(node_b, node_c, A, I, E)

print(solve_frame([element_1, element_2]))
