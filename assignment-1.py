import math
import matplotlib.pyplot as plt
import numpy as np

from frame_solver import FrameSystem, Node, FrameElement
from visualization import plot_element_fbd, plot_system_dofs, plot_system_deflection
from print_matrix import print_matrix, print_matrix_rounded, print_table

E = 200e9
G = 77e9

D = 100e-3
d = 90e-3

A = (D**2 - d**2) * math.pi / 4
I = (D**4 - d**4) * math.pi / 64

AS = 2 * A / math.pi


def build_structure(timoshenko=False):
    system = FrameSystem()

    # Build bottom left triangle
    node_a = system.create_node("A", 0, 0)
    node_a.fixed_joint()
    node_b = system.create_node("B", 0.25, 0.33)
    node_c = system.create_node("C", 0, 0.33)
    node_c.fixed_joint()

    if timoshenko:
        element_1 = system.create_element_timoshenko(node_a, node_b, A, I, E, G, AS)
        element_2 = system.create_element_timoshenko(node_c, node_b, A, I, E, G, AS)
    else:
        element_1 = system.create_element(node_a, node_b, A, I, E)
        element_2 = system.create_element(node_c, node_b, A, I, E)

    node_d = system.create_node("D", 0.375, 0.5)

    if timoshenko:
        element_3 = system.create_element_timoshenko(node_b, node_d, A, I, E, G, AS)
    else:
        element_3 = system.create_element(node_b, node_d, A, I, E)

    node_e = system.create_node("E", 0.75, 0.5)

    if timoshenko:
        element_4 = system.create_element_timoshenko(node_d, node_e, A, I, E, G, AS)
    else:
        element_4 = system.create_element(node_d, node_e, A, I, E)

    node_f = system.create_node("F", 0.75, 1)

    if timoshenko:
        element_5 = system.create_element_timoshenko(node_d, node_f, A, I, E, G, AS)
        element_6 = system.create_element_timoshenko(node_e, node_f, A, I, E, G, AS)
    else:
        element_5 = system.create_element(node_d, node_f, A, I, E)
        element_6 = system.create_element(node_e, node_f, A, I, E)

    node_g = system.create_node("G", 0.75, 1.5)

    if timoshenko:
        element_7 = system.create_element_timoshenko(node_f, node_g, A, I, E, G, AS)
    else:
        element_7 = system.create_element(node_f, node_g, A, I, E)

    return system


#
# Part 1
#

system = build_structure()

system.nodes["G"].force_x(-2e3)
system.nodes["E"].force_x(-2e3)

system.solve()


print("Part 1")

print("Tip Deflections")
print_matrix(system.dof_deflections[-3:], scale_coef=1e-3)
print()

print("Bottom Support Reaction:")
print_matrix(system.nodes["A"].reaction_load, scale_coef=1e3)
print("Top Support Reaction:")
print_matrix(system.nodes["C"].reaction_load, scale_coef=1e3)
print("Total support Reaction:")
print_matrix(
    system.nodes["A"].reaction_load + system.nodes["C"].reaction_load, scale_coef=1e3
)
print("Total System Reaction:")
print_matrix(sum(node.reaction_load for node in system.nodes.values()), scale_coef=1e3)
print()


def equilibrium_test(
    name: str, target_node: Node, nodes: list[Node], elements: list[FrameElement]
):
    print(f"Subsection Analysis around {name} (node {target_node.name})")
    total_force_reaction_moment = 0
    for reaction in nodes:
        load = reaction.reaction_load[:2]
        offset = reaction.position - target_node.position
        moment = np.cross(
            offset,
            load,
        )

        total_force_reaction_moment += moment

        print(f"Reaction Force at node {reaction.name}:")
        print_matrix(load)
        print(f"Position offset from node {target_node.name}:")
        print_matrix(offset)
        print(f"Moment due to node {reaction.name}:")
        print(moment)

    print(
        f"Sum of moments due to reaction forces: {total_force_reaction_moment * 1e-3:.4f}kNm"
    )

    print("Sum of Elemental global forces:")

    total_elemental_forces: np.ndarray = sum(
        (
            force
            for element in elements
            for force in (
                element.get_global_forces()[:3],
                element.get_global_forces()[3:],
            )
        ),
        start=np.array([0, 0, 0]),
    )

    print_matrix(total_elemental_forces, scale_coef=1e3)

    print(
        f"{total_elemental_forces[2] * 1e-3:.4f} + {total_force_reaction_moment * 1e-3:.4f} = {(total_elemental_forces[2] + total_force_reaction_moment) * 1e-3:.4f}"
    )


equilibrium_test(
    "C", system.nodes["B"], [system.nodes["A"], system.nodes["C"]], system.elements[:2]
)

equilibrium_test(
    "B",
    system.nodes["D"],
    [system.nodes["A"], system.nodes["C"]],
    system.elements[:3],
)


equilibrium_test(
    "A",
    system.nodes["F"],
    [system.nodes["A"], system.nodes["C"], system.nodes["E"]],
    system.elements[:6],
)

print("Max Element Normal Stresses")
for i, element in enumerate(system.elements):
    start_moment = abs(element.get_local_forces()[2])
    end_moment = abs(element.get_local_forces()[5])
    max_stress = (
        abs(element.get_axial_stress()) + max(start_moment, end_moment) * D / 2 / I
    )

    print(
        f"Element {i + 1}: {max_stress * 1e-6:.1f} MPa occuring at the {"start" if start_moment > end_moment else "end"}"
    )

print()

for i, element in enumerate(system.elements, start=1):
    print(f"Element {i} Assembly Matrix:")
    print_matrix_rounded(element.assembly_matrix)
fig, (axis1, axis2) = plt.subplots(1, 2)

plot_system_deflection(axis1, system)
plot_system_dofs(axis2, system, arrow_scale=0.05)

plt.show()


#
# Part 2
#

system = build_structure()

PRESSURE = 0.61e3

WIDTH = 3

UDL = PRESSURE * WIDTH

FOS = 2.5

system.elements[5].add_distributed_shear_load(lambda _: UDL)
system.elements[6].add_distributed_shear_load(lambda _: UDL)

system.solve()


print("Part 2")

print("Max Element Normal Stresses")

max_stresses = [
    abs(element.get_axial_stress())
    + max(abs(element.get_local_forces()[2]), abs(element.get_local_forces()[5]))
    * D
    / 2
    / I
    for element in system.elements
]

for i, stress in enumerate(max_stresses):
    print(f"Element {i + 1}: {stress * 1e-6:.1f} MPa")

YIELD_STRENGTH = 360e6

max_pressure = (350e6 / max(max_stresses)) * PRESSURE / FOS

max_speed = math.sqrt(max_pressure / 0.6)

print(
    f"Max allowable wind speed if Max stress is 350MPa with FOS of {FOS}: {max_speed:.0f}m/s ({max_speed * 3.6:.0f}km/h)"
)

system = build_structure()

udl_new = max_pressure * WIDTH


system.elements[5].add_distributed_shear_load(lambda _: udl_new)
system.elements[6].add_distributed_shear_load(lambda _: udl_new)

system.solve()

print("Tip Deflections")
print_matrix(system.elements[-1].get_local_deflections()[3:], scale_coef=1e-3)
print()

print("Bottom Support Reaction:")
print_matrix(system.nodes["A"].reaction_load, scale_coef=1e3)
print("Top Support Reaction:")
print_matrix(system.nodes["C"].reaction_load, scale_coef=1e3)

timoshenko_system = build_structure(timoshenko=True)

timoshenko_system.elements[5].add_distributed_shear_load(lambda _: udl_new)
timoshenko_system.elements[6].add_distributed_shear_load(lambda _: udl_new)

timoshenko_system.solve()

print("Tip Deflections (Timoshenko):")
print_matrix(
    timoshenko_system.elements[-1].get_local_deflections()[3:], scale_coef=1e-3
)
print()

print("Bottom Support Reaction (Timoshenko):")
print_matrix(timoshenko_system.nodes["A"].reaction_load, scale_coef=1e3)
print("Top Support Reaction (Timoshenko):")
print_matrix(timoshenko_system.nodes["C"].reaction_load, scale_coef=1e3)

print("Timoshenko vs Original Transverse Deflections")

timoshenko_transverse = np.array(
    [el.get_change_in_tranverse_deflection() for el in timoshenko_system.elements]
)

euler_transverse = np.array(
    [el.get_change_in_tranverse_deflection() for el in system.elements]
)

transverse_difference = timoshenko_transverse - euler_transverse

total = [["Element", "Timoshenko", "Euler", "Diff"]]
for i, (timo, euler, diff) in enumerate(
    zip(timoshenko_transverse, euler_transverse, transverse_difference), start=1
):
    total.append(
        [str(i), f"{timo * 1e3:.2f}", f"{euler * 1e3:.2f}", f"{diff * 1e3:.2f}"]
    )

print_table(total, suffix="x 1e-3")

print("Timoshenko vs Original Rotation Deflections")

timoshenko_rotation = np.array(
    [el.get_change_in_rotation_deflection() for el in timoshenko_system.elements]
)

euler_rotation = np.array(
    [el.get_change_in_rotation_deflection() for el in system.elements]
)

rotation_difference = timoshenko_rotation - euler_rotation

total = [["Element", "Timoshenko", "Euler", "Diff"]]
for i, (timo, euler, diff) in enumerate(
    zip(timoshenko_rotation, euler_rotation, rotation_difference), start=1
):
    total.append(
        [str(i), f"{timo * 1e3:.2f}", f"{euler * 1e3:.2f}", f"{diff * 1e3:.2f}"]
    )

print_table(total, suffix="x 1e-3")

print("Custom Design")

#
# Structural Modifications
#

modified_system = FrameSystem()

node_a = modified_system.create_node("A", 0, 0)
node_a.fixed_joint()
node_b = modified_system.create_node("B", 0, 0.33)
node_b.fixed_joint()

node_c = modified_system.create_node("C", 0.75, 0.5)
node_d = modified_system.create_node("D", 0.75, 0.5 + 0.33)

modified_system.create_element_timoshenko(node_a, node_c, A, I, E, G, AS)
modified_system.create_element_timoshenko(node_b, node_d, A, I, E, G, AS)
modified_system.create_element_timoshenko(node_c, node_d, A, I, E, G, AS)
modified_system.create_element_timoshenko(node_b, node_c, A, I, E, G, AS)

node_e = modified_system.create_node("E", 0.75, 1.5)

modified_system.create_element_timoshenko(node_d, node_e, A, I, E, G, AS)

modified_system.elements[3].add_distributed_shear_load(lambda _: UDL)
modified_system.elements[4].add_distributed_shear_load(lambda _: UDL)

modified_system.solve()

print("Max Element Normal Stresses")
max_stresses = [
    abs(element.get_axial_stress())
    + max(abs(element.get_local_forces()[2]), abs(element.get_local_forces()[5]))
    * D
    / 2
    / I
    for element in modified_system.elements
]

for i, stress in enumerate(max_stresses):
    print(f"Element {i + 1}: {stress * 1e-6:.1f} MPa")

max_pressure = (350e6 / max(max_stresses)) * PRESSURE / FOS

max_speed = math.sqrt(max_pressure / 0.6)

print(
    f"Max allowable wind speed if Max stress is 350MPa with FOS of {FOS} on modified system: {max_speed:.0f}m/s ({max_speed * 3.6:.0f}km/h)"
)

print(
    f"Total Original Element Length: {sum(element.length() for element in system.elements)}"
)

print(
    f"Total Modified Element Length: {sum(element.length() for element in modified_system.elements)}"
)
