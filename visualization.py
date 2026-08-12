from frame_solver import FrameElement, FrameSystem
from matplotlib.axes import Axes
import numpy as np

POINTS_PER_ELEMENT = 50


def plot_element(axes: Axes, element: FrameElement, label: str, deflection_scaling=20):
    t = np.linspace(0, 1, POINTS_PER_ELEMENT)

    undeflected = element.undeflected_position(t)

    axes.plot(undeflected[:, 0], undeflected[:, 1], "b-")

    deflected = element.deflected_position(t, deflection_scaling)

    axes.plot(deflected[:, 0], deflected[:, 1], "r--")


def plot_system(axes: Axes, system: FrameSystem, deflection_scaling=20):
    axes.set_title(f"System with deflection scaling of {deflection_scaling}x")
    axes.set_aspect("equal")
    for i, element in enumerate(system.elements):
        plot_element(axes, element, str(i), deflection_scaling=deflection_scaling)
