from frame_solver import FrameElement, FrameSystem
from matplotlib.axes import Axes
import numpy as np

POINTS_PER_ELEMENT = 50


def plot_element(axes: Axes, element: FrameElement, deflection_scaling=20):
    t = np.linspace(0, 1, POINTS_PER_ELEMENT)

    undeflected = element.undeflected_position(t)

    axes.plot(undeflected[:, 0], undeflected[:, 1], "b-")

    deflected = element.deflected_position(t, deflection_scaling)

    axes.plot(deflected[:, 0], deflected[:, 1], "r--")


def plot_system(axes: Axes, system: FrameSystem, deflection_scaling=20):
    axes.set_title(f"System with deflection scaling of {deflection_scaling}x")
    axes.set_aspect("equal")
    for element in system.elements:
        plot_element(axes, element, deflection_scaling=deflection_scaling)


ZERO_THRESHOLD = 1e-10


def format_number(x, decimals, max_len=None):
    if abs(x) > ZERO_THRESHOLD:
        return (
            f"{x:>{max_len}.{decimals}f}"
            if max_len is not None
            else f"{x:.{decimals}f}"
        )
    else:
        return f"{0:^{max_len}}" if max_len is not None else str(0)


def print_matrix(matrix, decimals=4, scale_coef=None):
    if scale_coef is None:
        max_val = np.max(np.abs(matrix))
        exp = np.floor(np.log10(max_val)) if max_val > 0 else 0
        scale_coef = 10**exp

    # Scale matrix values
    printed_matrix = matrix / scale_coef

    print(f"Scaling coefficient: {scale_coef:.0e}")
    max_len = max(len(format_number(x, decimals)) for x in printed_matrix.flatten())

    for row in printed_matrix:
        if isinstance(row, np.float64):
            print(f"|{format_number(row, decimals, max_len=max_len)}|")
        else:
            print(" ".join(format_number(x, decimals, max_len=max_len) for x in row))


def print_matrix_rounded(matrix):
    for row in matrix:
        print(" ".join(str(int(x)) for x in row))


def log_system_data(system: FrameSystem):
    print("Assembly Matrices:")
    for i, element in enumerate(system.elements):
        print(f"Element {i + 1}:")
        print_matrix_rounded(element.assembly_matrix)
    print()

    print("Global Stiffness Matrices:")
    for i, element in enumerate(system.elements):
        print(f"Element {i + 1}:")
        print_matrix(element.get_global_stiffness())
    print()

    print("Total Stiffness Matrix:")
    print_matrix(system.get_total_stiffness())
    print()

    print("Global Deflections:")
    print_matrix(system.dof_deflections, scale_coef=1e-3)
    print()

    print("Global Forces:")
    for i, element in enumerate(system.elements):
        print(f"Element {i + 1}:")
        print_matrix(element.get_global_forces(), scale_coef=1e3)
    print()
