"""
FEA Frame Solver: Alex Cutforth August 2026

Logic for generically representing and solving Frame systems
using Finite Element Analysis principles

NOTE: DOF ordering is not the same as is taught in ENME302
All lateral (X/Y) DOFs come first, followed by rotational DOFs
"""

import math
from collections.abc import Callable
import numpy as np


class SystemUnsolvedException(Exception):
    """Raised when the system is required to be solved, but is not yet"""


class Node:
    """
    Presents a nodal location in a FEA model
    """

    applied_x: float | None = 0
    applied_y: float | None = 0
    applied_moment: float | None = 0

    reaction_load: np.ndarray

    def __init__(
        self,
        name: str,
        position: np.ndarray,
    ):
        self.name = name
        self.position = position
        self.reaction_load = np.zeros(3)

    def force_x(self, force: float):
        """Add an applied force in the X axis"""
        self.applied_x = force

    def force_y(self, force: float):
        """Add an applied force in the Y axis"""
        self.applied_y = force

    def force_moment(self, force: float):
        """Add an applied moment"""
        self.applied_moment = force

    def fix_x(self):
        """Remove freedom of motion from the X axis"""
        self.applied_x = None

    def fix_y(self):
        """Remove freedom of motion from the Y axis"""
        self.applied_y = None

    def fix_rotation(self):
        """Remove freedom of rotational motion"""
        self.applied_moment = None

    def fixed_joint(self):
        """Removes all degrees of freedom from the node"""
        self.fix_x()
        self.fix_y()
        self.fix_rotation()

    def pin_joint(self):
        """Removes freedom of translational motion, with rotation remaining"""
        self.fix_x()
        self.fix_y()

    def x_slider_joint(self):
        """Removes freedom of motion in the Y axis"""
        self.fix_y()

    def y_slider_joint(self):
        """Removes freedom of motion in the X axis"""
        self.fix_x()


class FrameElement:
    """Represents a Frame element in an FEA model. Each frame connects two Nodes"""

    start: Node
    end: Node
    area: float
    second_moment_of_inertia: float
    elastic_modulus: float

    assembly_matrix: np.ndarray | None = None
    global_deflections: np.ndarray | None = None

    distributed_shear_loads: list[Callable[[np.ndarray], np.ndarray]]
    point_shear_loads: list[tuple[float, float]]

    distributed_axial_loads: list[Callable[[np.ndarray], np.ndarray]]
    point_axial_loads: list[tuple[float, float]]

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
        self.distributed_shear_loads = []
        self.point_shear_loads = []

        self.distributed_axial_loads = []
        self.point_axial_loads = []

    def alpha(self):
        """The angle in radians of the element from start to end, measured CCW from the X-axis"""
        vector = self.end.position - self.start.position

        return math.atan2(vector[1], vector[0])

    def length(self):
        """Returns the cartesian length of the element"""

        return np.linalg.norm(self.end.position - self.start.position)

    def add_distributed_shear_load(self, load: Callable[[np.ndarray], np.ndarray]):
        """
        Adds a distributed shear load across the element.
        Load function takes in parameter t (0 -> 1) and returns the force per length at each point
        """
        self.distributed_shear_loads.append(load)

    def add_point_shear_load(self, load: float, point: float):
        """Adds a point shear load at some point across the element (position measured between 0 and 1)"""
        self.point_shear_loads.append((load, point))

    def add_distributed_axial_load(self, load: Callable[[np.ndarray], np.ndarray]):
        """
        Adds a distributed axial load across the element.
        Load function takes in parameter t (0 -> 1) and returns the force per length at each point
        """
        self.distributed_axial_loads.append(load)

    def add_point_axial_load(self, load: float, point: float):
        """Adds a point axial load at some point across the element (position measured between 0 and 1)"""
        self.point_axial_loads.append((load, point))

    def get_local_equivalent_distributed_load(self):
        """
        Returns the local equivalent forces due to any added distributed loads
        """
        if (
            len(self.distributed_shear_loads) == 0
            and len(self.distributed_axial_loads) == 0
        ):
            return np.zeros(6)

        t = np.linspace(0, 1, 100)

        # Sum all distributed loads for each t
        shear_load = sum(load(t) for load in self.distributed_shear_loads)
        axial_load = sum(load(t) for load in self.distributed_axial_loads)

        shear_integrand = self.get_lateral_shape_functions(t) * shear_load
        axial_integrand = self.get_axial_shape_functions(t) * axial_load

        # Integrate the distributed load for each shape function
        f_eq: np.ndarray = np.trapezoid(shear_integrand + axial_integrand, t, axis=1)  # type: ignore[assignment]

        # Unnormalize based on element length
        return self.length() * f_eq

    def get_local_equivalent_point_load(self):
        """
        Returns the local equivalent forces due to any added point loads
        """
        if len(self.point_shear_loads) == 0 and len(self.point_axial_loads) == 0:
            return np.zeros(6)

        f_shear = sum(
            load * self.get_lateral_shape_functions(point)
            for load, point in self.point_shear_loads
        )

        f_axial = sum(
            load * self.get_axial_shape_functions(point)
            for load, point in self.point_axial_loads
        )

        return f_shear + f_axial

    def get_global_equivalent_loading(self):
        """
        Returns the global forces due to any extra element based such as distributed or point loads
        """

        f_eq: np.ndarray = (
            self.get_local_equivalent_distributed_load()
            + self.get_local_equivalent_point_load()
        )

        return self.get_transformation_matrix().T @ f_eq

    def get_lateral_shape_functions(self, t: np.ndarray | float):
        """
        Returns the value of the lateral shape functions for a given t (array or float)

        Resulting array has 6 values, with indices 0 and 3 being all zero
        """
        l = np.linalg.norm(self.end.position - self.start.position)

        # Shape functions are only relevant for the Beam components of the Frame
        # To simplify downstream logic, n0 and n3 are set to zero to represent the isolation between axial force and lateral deflection

        z = np.zeros_like(t)

        n1 = 1 - 3 * t**2 + 2 * t**3
        n2 = t**3 * l - 2 * t**2 * l + t * l
        n3 = 3 * t**2 - 2 * t**3
        n4 = t**3 * l - t**2 * l

        return np.array([z, n1, n2, z, n3, n4])

    def get_axial_shape_functions(self, t: np.ndarray | float):
        """
        Returns the value of the axial shape functions for a given t (array or float)

        Resulting array has 6 values, with indices 1, 2, 4, and 5 being all zero
        """

        z = np.zeros_like(t)

        return np.array([1 - t, z, z, t, z, z])

    def undeflected_position(self, t: np.ndarray) -> np.ndarray:
        """
        Returns the undeflected coordinate for a given t (0 -> 1)
        """
        return self.start.position + t[:, None] * (
            self.end.position - self.start.position
        )

    def deflected_position(self, t: np.ndarray, scale) -> np.ndarray:
        """Returns the deflected coordinate for a given t (0 -> 1)"""
        d_e = self.get_local_deflections()

        axial_deflection = d_e.dot(self.get_axial_shape_functions(t))
        transverse_deflection = d_e.dot(self.get_lateral_shape_functions(t))

        deflections_x = axial_deflection * math.cos(
            self.alpha()
        ) - transverse_deflection * math.sin(self.alpha())

        deflections_y = axial_deflection * math.sin(
            self.alpha()
        ) + transverse_deflection * math.cos(self.alpha())

        deflections = np.column_stack((deflections_x, deflections_y))

        return self.undeflected_position(t) + deflections * scale

    def get_local_stiffness(self):
        """
        Returns the local stiffness matrix for the element (6x6)
        """
        l = np.linalg.norm(self.end.position - self.start.position)

        beta = self.area * l**2 / self.second_moment_of_inertia

        coef = self.elastic_modulus * self.second_moment_of_inertia / (l**3)

        return coef * np.array(
            [
                [beta, 0, 0, -beta, 0, 0],
                [0, 12, 6 * l, 0, -12, 6 * l],
                [0, 6 * l, 4 * l**2, 0, -6 * l, 2 * l**2],
                [-beta, 0, 0, beta, 0, 0],
                [0, -12, -6 * l, 0, 12, -6 * l],
                [0, 6 * l, 2 * l**2, 0, -6 * l, 4 * l**2],
            ]
        )

    def get_transformation_matrix(self):
        """Returns the rotational transformation matrix for the element (6x6)"""
        s = math.sin(self.alpha())
        c = math.cos(self.alpha())

        inner = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        return np.block([[inner, np.zeros_like(inner)], [np.zeros_like(inner), inner]])

    def get_stiffness(self):
        """Gets the global aligned stiffness matrix for the element (6x6)"""
        _lambda = self.get_transformation_matrix()

        return _lambda.T @ self.get_local_stiffness() @ _lambda

    def get_global_stiffness(self):
        """
        Gets the global nodal stiffness matrix for the element

        This function will raise an exception if the system is not yet solved
        """
        if self.assembly_matrix is None:
            raise SystemUnsolvedException()

        return self.assembly_matrix @ self.get_stiffness() @ self.assembly_matrix.T

    def get_local_deflections(self):
        """
        Gets the deflection of the element in local coordinates

        This function will raise an exception if the system is not yet solved
        """
        if self.global_deflections is None:
            raise SystemUnsolvedException()

        return self.get_transformation_matrix() @ self.global_deflections

    def get_local_forces(self):
        """
        Gets the resulting forces of the element in local coordinates

        This function will raise an exception if the system is not yet solved
        """
        return self.get_local_stiffness() @ self.get_local_deflections()

    def get_global_forces(self):
        """
        Gets the resulting forces of the element in global coordinates

        This function will raise an exception if the system is not yet solved
        """

        return self.get_transformation_matrix().T @ self.get_local_forces()


class FrameSystem:
    """Represents a finite element analysis system using frame elements"""

    nodes: dict[str, Node] = {}
    elements: list[FrameElement] = []
    weld_pairs: list[tuple[FrameElement, FrameElement]] = []

    nodal_forces: np.ndarray
    dof_deflections: np.ndarray

    x_dof_indices: dict[str, int]
    y_dof_indices: dict[str, int]
    start_moment_indices: dict[int, int]
    end_moment_indices: dict[int, int]

    def __init__(self):
        self.x_dof_indices = {}
        self.y_dof_indices = {}
        self.start_moment_indices = {}
        self.end_moment_indices = {}

    def create_node(self, name: str, x: float, y: float) -> Node:
        """Creates a new node at a given coordinate"""
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
        """Creates a new element connecting two nodes with a given A, I, E"""
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
        """Fixes two elements together rotationaly to prevent generation of extra rotational DOFs"""
        self.weld_pairs.append((element_1, element_2))

    def get_total_stiffness(self):
        """Returns the total global stiffness matrix for the system"""
        return sum(element.get_global_stiffness() for element in self.elements)

    def solve(self):
        """Finalizes the structure of the system and calculates deflections and forces"""

        # To simplify logic, nodal force ordering has all lateral DOFs first,
        # with rotational forces at the end
        self.nodal_forces = np.array([])

        # Lateral DOF indices are mapped by the Node Name
        # Rotational DOF indices are mapped by the element index
        # In a worst case, every element may have its own two moment DOFs

        # For each node, apply the nodal forces to the total Q array
        for node in self.nodes.values():
            # A value of None represents constrainted motion
            if node.applied_x is not None:
                self.nodal_forces = np.append(self.nodal_forces, node.applied_x)
                self.x_dof_indices[node.name] = len(self.nodal_forces) - 1

            if node.applied_y is not None:
                self.nodal_forces = np.append(self.nodal_forces, node.applied_y)
                self.y_dof_indices[node.name] = len(self.nodal_forces) - 1

        # Applying rotational forces is difficult, as duplicate degrees of freedom have to be
        # added as every joint between elements is assumed to be a pin,
        # unless they are marked as welded
        for i, element in enumerate(self.elements):
            # The first pass through elements generates rotational DOFs

            start = element.start
            end = element.end

            # Find all elements welded to this element
            welds = [
                x for e in self.weld_pairs for x in e if element in e and element != x
            ]

            if start.applied_moment is not None:
                existing_index = None

                # If the node at the start is not fixed, search for any existing degrees of freedom
                # linked to elements welded at the start
                for e in welds:
                    index = self.elements.index(e)

                    if start == e.start and index in self.start_moment_indices:
                        existing_index = self.start_moment_indices[index]

                    if start == e.end and index in self.end_moment_indices:
                        existing_index = self.end_moment_indices[index]

                # If no DOF yet exists, create one
                if existing_index is None:
                    self.nodal_forces = np.append(
                        self.nodal_forces, start.applied_moment
                    )
                    self.start_moment_indices[i] = len(self.nodal_forces) - 1
                else:
                    self.start_moment_indices[i] = existing_index

            if end.applied_moment is not None:
                existing_index = None

                # If the node at the end is not fixed, search for any existing degrees of freedom
                # linked to elements welded at the end
                for e in welds:
                    index = self.elements.index(e)
                    if end == e.start and index in self.start_moment_indices:
                        existing_index = self.start_moment_indices[index]

                    if end == e.end and index in self.end_moment_indices:
                        existing_index = self.end_moment_indices[index]

                # If no DOF yet exists, create one
                if existing_index is None:
                    self.nodal_forces = np.append(self.nodal_forces, end.applied_moment)
                    self.end_moment_indices[i] = len(self.nodal_forces) - 1
                else:
                    self.end_moment_indices[i] = existing_index

        for i, element in enumerate(self.elements):
            # The second pass through elements generates assembly matrices

            start = element.start
            end = element.end

            assembly_matrix = np.zeros((len(self.nodal_forces), 6))

            if start.name in self.x_dof_indices:
                assembly_matrix[self.x_dof_indices[start.name], 0] = 1

            if start.name in self.y_dof_indices:
                assembly_matrix[self.y_dof_indices[start.name], 1] = 1

            if i in self.start_moment_indices:
                assembly_matrix[self.start_moment_indices[i], 2] = 1

            if end.name in self.x_dof_indices:
                assembly_matrix[self.x_dof_indices[end.name], 3] = 1

            if end.name in self.y_dof_indices:
                assembly_matrix[self.y_dof_indices[end.name], 4] = 1

            if i in self.end_moment_indices:
                assembly_matrix[self.end_moment_indices[i], 5] = 1

            element.assembly_matrix = assembly_matrix

            # After the assembly matrix has been generated, calculate the equivalent nodal
            # loading due to point and distributed loads, and append to the total nodal forces
            self.nodal_forces += (
                element.assembly_matrix @ element.get_global_equivalent_loading()
            )

        self.dof_deflections = np.linalg.solve(
            self.get_total_stiffness(), self.nodal_forces
        )

        for element in self.elements:
            element.global_deflections = (
                element.assembly_matrix.T @ self.dof_deflections
            )

        for node in self.nodes.values():
            for element in self.elements:
                if element.start == node:
                    node.reaction_load += (
                        element.get_global_forces()[:3]
                        - element.get_global_equivalent_loading()[:3]
                    )
                if element.end == node:
                    node.reaction_load += (
                        element.get_global_forces()[3:]
                        - element.get_global_equivalent_loading()[3:]
                    )
