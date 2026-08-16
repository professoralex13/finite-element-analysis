import numpy as np
import matplotlib.pyplot as plt
from frame_solver import FrameSystem
from visualization import (
    plot_system_deflection,
    plot_system_dofs,
    log_system_data,
)

from print_matrix import (
    print_matrix,
    print_matrix_rounded,
)

A = 2e-4
I = 5e-6
E = 200e9

system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_c = system.create_node("C", 4, 0)
node_c.x_slider_joint()

node_b = system.create_node("B", 4, 3)
node_b.force_x(20e3)
node_b.force_y(40e3)

element_1 = system.create_element(node_a, node_c, A, I, E)
element_1.add_distributed_shear_load(lambda t: -25e3 * np.ones_like(t))

element_2 = system.create_element(node_a, node_b, A, I, E)
element_3 = system.create_element(node_c, node_b, A, I, E)

system.solve()

print("c)")
for i, element in enumerate(system.elements):
    print(f"Element {i + 1} Assembly Matrix:")
    print_matrix_rounded(element.assembly_matrix)
print()

print("d)")
for i, element in enumerate(system.elements):
    print(f"Element {i + 1} Global stiffness matrix:")
    print_matrix(element.get_global_stiffness())
print()

print("e)")
print("element 1 equivalent forcing vector:")
print_matrix(element_1.get_local_equivalent_distributed_load(), scale_coef=1e3)
print("Overall global forcing vector Q:")
print_matrix(system.nodal_forces, scale_coef=1e3)
print()

print("f)")
print_matrix(system.dof_deflections, scale_coef=1e-3)
print()

fig, (axis1, axis2) = plt.subplots(1, 2)

plot_system_deflection(
    axis1, system, title="Sketch from Question g)", deflection_scaling=50
)

plot_system_dofs(axis2, system, title="Question a)")

print("h)")
print("Reaction Load at A:")
print_matrix(node_a.reaction_load, scale_coef=1e3)
print("Reaction Load at C:")
print_matrix(node_c.reaction_load, scale_coef=1e3)
print()

print("i)")
print("Local Forces in Element 2")
print_matrix(element_2.get_local_forces(), scale_coef=1e3)
print(
    f"Max Moment: {element_2.get_local_forces()[5] * 1e-3:.3}kNm, which occurs at the end of Element 2"
)
print()

print("j)")
print(
    f"Horizontal Deflection at midspan of element 1: {element_1.deflection_vector(0.5)[0][0] * 1e3:.3}mm"
)

plt.show()
