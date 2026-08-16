from matplotlib.patches import Arc, RegularPolygon

from frame_solver import FrameElement, FrameSystem
from matplotlib.axes import Axes
import numpy as np

from print_matrix import print_matrix, print_matrix_rounded

POINTS_PER_ELEMENT = 50


def plot_element(
    axes: Axes, element: FrameElement, deflection_scaling=20, show_deflections=True
):
    t = np.linspace(0, 1, POINTS_PER_ELEMENT)

    undeflected = element.undeflected_position(t)

    axes.plot(undeflected[:, 0], undeflected[:, 1], "b-")

    if show_deflections:
        deflected = undeflected + element.deflection_vector(t) * deflection_scaling

        axes.plot(deflected[:, 0], deflected[:, 1], "r--")


def plot_system_deflection(
    axes: Axes, system: FrameSystem, title: str | None = None, deflection_scaling=20
):
    axes.set_title(title or f"System with deflection scaling of {deflection_scaling}x")
    axes.set_aspect("equal")
    for element in system.elements:
        plot_element(axes, element, deflection_scaling=deflection_scaling)


def plot_element_fbd(
    axes: Axes,
    system: FrameSystem,
    element: FrameElement,
    arrow_scale: float = 0.25,
):
    index = system.elements.index(element)

    axes.set_title(f"Element {index + 1} FBD")
    axes.set_aspect("equal")
    axes.axis("off")

    plot_element(axes, element, show_deflections=False)

    midpoint = (element.start.position + element.end.position) / 2

    x_e = element.end.position - element.start.position

    x_e /= np.linalg.norm(x_e)

    y_e = np.array([-x_e[1], x_e[0]])

    text_pos = midpoint - y_e * arrow_scale * 0.5

    axes_pos = midpoint + y_e * arrow_scale * 0.5

    axes.arrow(
        axes_pos[0],
        axes_pos[1],
        (y_e * arrow_scale)[0],
        (y_e * arrow_scale)[1],
        width=arrow_scale * 0.05,
        head_width=arrow_scale * 0.2,
        head_length=arrow_scale * 0.3,
        fc="black",
        ec="black",
    )

    text_angle = (
        element.alpha()
        if -np.pi / 2 < element.alpha() < np.pi / 2
        else element.alpha() + np.pi
    )

    axes.text(
        (axes_pos + y_e * arrow_scale)[0] + arrow_scale * 0.3,
        (axes_pos + y_e * arrow_scale)[1],
        "y_e",
        fontsize=10,
        color="black",
    )

    axes.text(
        (axes_pos + x_e * arrow_scale)[0] + arrow_scale * 0.3,
        (axes_pos + x_e * arrow_scale)[1],
        "x_e",
        fontsize=10,
        color="black",
    )

    axes.arrow(
        axes_pos[0],
        axes_pos[1],
        (x_e * arrow_scale)[0],
        (x_e * arrow_scale)[1],
        width=arrow_scale * 0.05,
        head_width=arrow_scale * 0.2,
        head_length=arrow_scale * 0.3,
        fc="black",
        ec="black",
    )

    box_properties = dict(
        boxstyle="square,pad=0.25",
        facecolor="none",
        edgecolor="black",
        linewidth=1,
    )

    axes.text(
        text_pos[0],
        text_pos[1],
        str(index + 1),
        fontsize=10,
        fontweight="bold",
        color="black",
        horizontalalignment="right",
        verticalalignment="center",
        bbox=box_properties,
        rotation=np.degrees(text_angle),
        rotation_mode="anchor",
    )

    axes.text(
        text_pos[0],
        text_pos[1],
        f"   α = {np.degrees(element.alpha()):.1f}°",
        fontsize=10,
        fontweight="bold",
        color="black",
        horizontalalignment="left",
        verticalalignment="center",
        rotation=np.degrees(text_angle),
        rotation_mode="anchor",
    )

    draw_dof_arrow(
        axes, element.start.position, np.array([1.0, 0.0]), arrow_scale, "D1"
    )
    draw_dof_arrow(
        axes, element.start.position, np.array([0.0, 1.0]), arrow_scale, "D2"
    )
    draw_rot_dof_arrow(axes, element.start.position, arrow_scale, arrow_scale, "D3")

    draw_dof_arrow(axes, element.end.position, np.array([1.0, 0.0]), arrow_scale, "D4")
    draw_dof_arrow(axes, element.end.position, np.array([0.0, 1.0]), arrow_scale, "D5")
    draw_rot_dof_arrow(axes, element.end.position, arrow_scale, arrow_scale, "D6")


def plot_system_dofs(
    axes: Axes, system: FrameSystem, title: str | None = None, arrow_scale: float = 0.25
):
    axes.set_title(title or "System DOFS")
    axes.set_aspect("equal")

    for i, element in enumerate(system.elements):
        plot_element(axes, element, show_deflections=False)

        midpoint = (element.start.position + element.end.position) / 2

        x_e = element.end.position - element.start.position

        x_e /= np.linalg.norm(x_e)

        y_e = np.array([-x_e[1], x_e[0]])

        text_pos = midpoint - y_e * arrow_scale * 0.5

        box_properties = dict(
            boxstyle="square,pad=0.25",
            facecolor="none",
            edgecolor="black",
            linewidth=1,
        )

        text_angle = (
            element.alpha()
            if -np.pi / 2 < element.alpha() < np.pi / 2
            else element.alpha() + np.pi
        )

        axes.text(
            text_pos[0],
            text_pos[1],
            str(i + 1),
            fontsize=10,
            fontweight="bold",
            color="black",
            horizontalalignment="right",
            verticalalignment="center",
            bbox=box_properties,
            rotation=np.degrees(text_angle),
            rotation_mode="anchor",
        )

    rot_node_occurances = {}

    for i in range(len(system.nodal_forces)):
        x_node = next(
            (
                system.nodes[node]
                for node, index in system.x_dof_indices.items()
                if i == index
            ),
            None,
        )

        y_node = next(
            (
                system.nodes[node]
                for node, index in system.y_dof_indices.items()
                if i == index
            ),
            None,
        )

        rot_node = next(
            (
                system.elements[el_index].start
                for el_index, dof_index in system.start_moment_indices.items()
                if i == dof_index
            ),
            None,
        ) or next(
            (
                system.elements[el_index].end
                for el_index, dof_index in system.end_moment_indices.items()
                if i == dof_index
            ),
            None,
        )

        if x_node:
            draw_dof_arrow(
                axes, x_node.position, np.array([1, 0]), arrow_scale, f"q{i+1}"
            )

        if y_node:
            draw_dof_arrow(
                axes, y_node.position, np.array([0, 1]), arrow_scale, f"q{i+1}"
            )

        if rot_node:
            if rot_node.name in rot_node_occurances:
                num_occurances = rot_node_occurances[rot_node.name]
                rot_node_occurances[rot_node.name] = num_occurances + 1
            else:
                num_occurances = 0
                rot_node_occurances[rot_node.name] = 1

            radius = arrow_scale * (1 + num_occurances * 0.5)

            draw_rot_dof_arrow(
                axes, rot_node.position, radius, arrow_scale, f"q{i + 1}"
            )


def draw_dof_arrow(axes: Axes, pos, direction: np.ndarray, scale, label: str):
    axes.arrow(
        pos[0],
        pos[1],
        (direction * scale)[0],
        (direction * scale)[1],
        width=scale * 0.05,
        head_width=scale * 0.2,
        head_length=scale * 0.3,
        fc="red",
        ec="red",
    )

    axes.text(
        (pos + direction * scale)[0] + scale * 0.25,
        (pos + direction * scale)[1],
        label,
        fontsize=24,
        color="red",
    )


def draw_rot_dof_arrow(axes: Axes, center, radius, scale, label: str):
    draw_circle_arrow(
        axes,
        radius,
        scale * 0.1,
        center[0],
        center[1],
    )

    axes.text(
        center[0] + radius * 0.1,
        center[1] - radius * 1,
        label,
        fontsize=24,
        color="red",
    )


def draw_circle_arrow(
    ax, radius, head_radius, centX, centY, angle_=90, theta2_=180, color_="red"
):
    """https://stackoverflow.com/questions/37512502/how-to-make-arrow-that-loops-in-matplotlib"""
    # ========Line
    arc = Arc(
        (centX, centY),
        radius * 2,
        radius * 2,
        angle=angle_,
        theta1=0,
        theta2=theta2_,
        capstyle="round",
        linestyle="-",
        lw=3,
        color=color_,
    )
    ax.add_patch(arc)

    # ========Create the arrow head
    endX = centX + (radius) * np.cos(
        np.radians(theta2_ + angle_)
    )  # Do trig to determine end position
    endY = centY + (radius) * np.sin(np.radians(theta2_ + angle_))

    ax.add_patch(  # Create triangle as arrow head
        RegularPolygon(
            (endX, endY),  # (x,y)
            3,  # number of vertices
            radius=head_radius,  # radius
            orientation=np.radians(angle_ + theta2_),  # orientation
            color=color_,
        )
    )


def log_system_data(system: FrameSystem):
    print("DOF Ordering:")
    for key, value in system.x_dof_indices.items():
        print(f"q{value + 1} -> Node {key} X")

    for key, value in system.y_dof_indices.items():
        print(f"q{value + 1} -> Node {key} Y")

    for key, value in system.start_moment_indices.items():
        node = system.elements[key].start.name
        print(f"q{value + 1} -> Node {node} Rot (Element {key + 1} Start)")

    for key, value in system.end_moment_indices.items():
        node = system.elements[key].end.name
        print(f"q{value + 1} -> Node {node} Rot (Element {key + 1} End)")

    print()
    print("Assembly Matrices (Element Directions along X, Global DOFs along Y):")
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

    print("DOF Forces:")
    print_matrix(system.nodal_forces, scale_coef=1e3)
    print()

    print("Global Deflections:")
    print_matrix(system.dof_deflections, scale_coef=1e-3)
    print()

    print("Nodal Deflections (Local Axes):")
    for i, element in enumerate(system.elements):
        print(f"Element {i + 1}:")
        print(f"Start (Node {element.start.name}):")
        print_matrix(element.get_local_deflections()[:3], scale_coef=1e-3)
        print(f"End (Node {element.end.name}):")
        print_matrix(element.get_local_deflections()[3:], scale_coef=1e-3)
    print()

    print("Nodal Reactions (Global Axes):")
    for name, node in system.nodes.items():
        print(f"(Node {name}):")
        print_matrix(node.reaction_load, scale_coef=1e3)
    print()

    print("Element Forces (Global Axes):")
    for i, element in enumerate(system.elements):
        print(f"Element {i + 1}:")
        print(f"Start (Node {element.start.name}):")
        print_matrix(element.get_global_forces()[:3], scale_coef=1e3)
        print(f"End (Node {element.end.name}):")
        print_matrix(element.get_global_forces()[3:], scale_coef=1e3)
    print()

    print("Element Forces (Local Axes):")
    for i, element in enumerate(system.elements):
        print(f"Element {i + 1}:")
        print(f"Start (Node {element.start.name}):")
        print_matrix(element.get_local_forces()[:3], scale_coef=1e3)
        print(f"End (Node {element.end.name}):")
        print_matrix(element.get_local_forces()[3:], scale_coef=1e3)
    print()
