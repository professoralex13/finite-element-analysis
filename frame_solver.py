import numpy as np
import math
from collections.abc import Callable


class Node:
    applied_x: float | None = 0
    applied_y: float | None = 0
    applied_moment: float | None = 0

    def __init__(
        self,
        name: str,
        position: np.ndarray,
    ):
        self.name = name
        self.position = position

    def force_x(self, force: float):
        self.applied_x = force

    def force_y(self, force: float):
        self.applied_y = force

    def force_moment(self, force: float):
        self.applied_moment = force

    def fix_x(self):
        self.applied_x = None

    def fix_y(self):
        self.applied_y = None

    def fix_rotation(self):
        self.applied_moment = None

    def fixed_joint(self):
        self.fix_x()
        self.fix_y()
        self.fix_rotation()

    def pin_joint(self):
        self.fix_x()
        self.fix_y()

    def x_slider_joint(self):
        self.fix_y()

    def y_slider_joint(self):
        self.fix_x()


class FrameElement:
    start: Node
    end: Node
    area: float
    second_moment_of_inertia: float
    elastic_modulus: float

    assembly_matrix: np.ndarray | None = None
    global_deflections: np.ndarray | None = None

    distributed_loads: list[Callable[[np.ndarray], np.ndarray]]
    point_loads: list[tuple[float, float]]

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
        self.distributed_loads = []
        self.point_loads = []

    def alpha(self):
        vector = self.end.position - self.start.position

        return math.atan2(vector[1], vector[0])

    def add_distributed_load(self, load: Callable[[np.ndarray], np.ndarray]):
        self.distributed_loads.append(load)

    def add_point_load(self, load: float, point: float):
        self.point_loads.append((load, point))

    def get_local_equivalent_distributed_load(self):
        if len(self.distributed_loads) == 0:
            return np.zeros(6)

        t = np.linspace(0, 1, 100)

        load = sum(load(t) for load in self.distributed_loads)

        integrand = self.get_shape_functions(t) * load

        f_eq: np.ndarray = np.trapezoid(integrand, t, axis=1)  # type: ignore[assignment]

        L = np.linalg.norm(self.end.position - self.start.position)

        return L * np.array([0, f_eq[0], f_eq[1], 0, f_eq[2], f_eq[3]])

    def get_local_equivalent_point_load(self):
        if len(self.point_loads) == 0:
            return np.zeros(6)

        f_eq: np.ndarray = sum(
            load * self.get_shape_functions(point) for load, point in self.point_loads
        )  # type: ignore[assignment]

        return np.array([0, f_eq[0], f_eq[1], 0, f_eq[2], f_eq[3]])

    def get_nodal_equivalent_loading(self):
        f_eq = (
            self.get_local_equivalent_distributed_load()
            + self.get_local_equivalent_point_load()
        )

        return self.assembly_matrix @ (self.get_transformation_matrix().T @ f_eq)

    def get_shape_functions(self, t: np.ndarray | float):
        L = np.linalg.norm(self.end.position - self.start.position)

        N_1 = 1 - 3 * t**2 + 2 * t**3
        N_2 = t**3 * L - 2 * t**2 * L + t * L
        N_3 = 3 * t**2 - 2 * t**3
        N_4 = t**3 * L - t**2 * L

        return np.array([N_1, N_2, N_3, N_4])

    def undeflected_position(self, t: np.ndarray) -> np.ndarray:
        return self.start.position + t[:, None] * (
            self.end.position - self.start.position
        )

    def deflected_position(self, t: np.ndarray, scale) -> np.ndarray:
        d_e = self.get_local_deflections()

        axial_deflection = (1 - t) * d_e[0] + t * d_e[3]

        N = self.get_shape_functions(t)

        transverse_deflection = d_e[[1, 2, 4, 5]].dot(N)

        deflections_x = axial_deflection * math.cos(
            self.alpha()
        ) - transverse_deflection * math.sin(self.alpha())

        deflections_y = axial_deflection * math.sin(
            self.alpha()
        ) + transverse_deflection * math.cos(self.alpha())

        deflections = np.column_stack((deflections_x, deflections_y))

        return self.undeflected_position(t) + deflections * scale

    def get_local_stiffness(self):
        L = np.linalg.norm(self.end.position - self.start.position)

        beta = self.area * L**2 / self.second_moment_of_inertia

        coef = self.elastic_modulus * self.second_moment_of_inertia / (L**3)

        return coef * np.array(
            [
                [beta, 0, 0, -beta, 0, 0],
                [0, 12, 6 * L, 0, -12, 6 * L],
                [0, 6 * L, 4 * L**2, 0, -6 * L, 2 * L**2],
                [-beta, 0, 0, beta, 0, 0],
                [0, -12, -6 * L, 0, 12, -6 * L],
                [0, 6 * L, 2 * L**2, 0, -6 * L, 4 * L**2],
            ]
        )

    def get_transformation_matrix(self):
        s = math.sin(self.alpha())
        c = math.cos(self.alpha())

        inner = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        return np.block([[inner, np.zeros_like(inner)], [np.zeros_like(inner), inner]])

    def get_stiffness(self):
        _lambda = self.get_transformation_matrix()
        return _lambda.T @ self.get_local_stiffness() @ _lambda

    def get_global_stiffness(self):
        if self.assembly_matrix is None:
            raise Exception(
                "Frame system must be solved before accessing element global stiffness"
            )

        return self.assembly_matrix @ self.get_stiffness() @ self.assembly_matrix.T

    def get_local_deflections(self):
        if self.global_deflections is None:
            raise Exception(
                "Frame system must be solved before accessing element global deflections"
            )

        return self.get_transformation_matrix() @ self.global_deflections

    def get_local_forces(self):
        return self.get_local_stiffness() @ self.get_local_deflections()

    def get_global_forces(self):
        return self.get_transformation_matrix().T @ self.get_local_forces()


class FrameSystem:
    nodes: dict[str, Node] = {}
    elements: list[FrameElement] = []
    weld_pairs: list[tuple[FrameElement, FrameElement]] = []

    dof_deflections: np.ndarray

    def create_node(self, name: str, x: float, y: float) -> Node:
        node = Node(name, np.array([x, y]))

        self.nodes[name] = node

        return node

    def create_element(
        self,
        node_1: Node,
        node_2: Node,
        area: float,
        moment_of_inertia: float,
        elastic_modulus: float,
    ) -> FrameElement:
        element = FrameElement(
            node_1,
            node_2,
            area,
            moment_of_inertia,
            elastic_modulus,
        )

        self.elements.append(element)

        return element

    def weld_elements(self, element_1: FrameElement, element_2: FrameElement):
        self.weld_pairs.append((element_1, element_2))

    def solve(self):
        Q = np.array([])

        x_indices = {}
        y_indices = {}
        start_moment_indices = {}
        end_moment_indices = {}

        for node in self.nodes.values():
            if node.applied_x is not None:
                Q = np.append(Q, node.applied_x)
                x_indices[node.name] = len(Q) - 1

            if node.applied_y is not None:
                Q = np.append(Q, node.applied_y)
                y_indices[node.name] = len(Q) - 1

        for i, element in enumerate(self.elements):
            start = element.start
            end = element.end

            welds = [
                x for e in self.weld_pairs for x in e if element in e and element != x
            ]

            if start.applied_moment is not None:
                existing_index = None
                for e in welds:
                    index = self.elements.index(e)
                    if start == e.start and index in start_moment_indices:
                        existing_index = start_moment_indices[index]

                    if start == e.end and index in end_moment_indices:
                        existing_index = end_moment_indices[index]

                if existing_index is None:
                    Q = np.append(Q, start.applied_moment)
                    start_moment_indices[i] = len(Q) - 1
                else:
                    start_moment_indices[i] = existing_index

            if end.applied_moment is not None:
                existing_index = None
                for e in welds:
                    index = self.elements.index(e)
                    if end == e.start and index in start_moment_indices:
                        existing_index = start_moment_indices[index]

                    if end == e.end and index in end_moment_indices:
                        existing_index = end_moment_indices[index]

                if existing_index is None:
                    Q = np.append(Q, end.applied_moment)
                    end_moment_indices[i] = len(Q) - 1
                else:
                    end_moment_indices[i] = existing_index

        for i, element in enumerate(self.elements):
            start = element.start
            end = element.end

            A = np.zeros((len(Q), 6))

            if start.name in x_indices:
                A[x_indices[start.name], 0] = 1

            if start.name in y_indices:
                A[y_indices[start.name], 1] = 1

            if i in start_moment_indices:
                A[start_moment_indices[i], 2] = 1

            if end.name in x_indices:
                A[x_indices[end.name], 3] = 1

            if end.name in y_indices:
                A[y_indices[end.name], 4] = 1

            if i in end_moment_indices:
                A[end_moment_indices[i], 5] = 1

            element.assembly_matrix = A

        for element in self.elements:
            Q += element.get_nodal_equivalent_loading()

        K_global = sum(element.get_global_stiffness() for element in self.elements)

        self.dof_deflections = np.linalg.solve(K_global, Q)

        for element in self.elements:
            element.global_deflections = (
                element.assembly_matrix.T @ self.dof_deflections
            )
