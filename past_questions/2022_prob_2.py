import numpy as np
import matplotlib.pyplot as plt
from frame_solver import FrameSystem
from visualization import plot_element_fbd, plot_system_dofs, plot_system_deflection
from print_matrix import print_matrix, print_matrix_rounded

A = 3e-4
I = 4.5e-6
E = 200e9


system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_b = system.create_node("B", 1.5, 0)

node_c = system.create_node(
    "C", 1.5 + 2 * np.cos(np.radians(15)), 2 * np.sin(np.radians(15))
)

node_d = system.create_node(
    "D", 1.5 - 2.5 * np.cos(np.radians(80)), -2.5 * np.sin(np.radians(80))
)
node_d.fixed_joint()

element_1 = system.create_element(node_a, node_b, A, I, E)
element_1.add_distributed_y_load(lambda _: -5e3)

element_2 = system.create_element(node_b, node_c, A, I, E)
element_2.add_distributed_y_load(lambda _: -5e3)

element_3 = system.create_element(node_b, node_d, A, I, E)

system.solve()

fig1, (axis1, axis2) = plt.subplots(1, 2)
fig2, (axis3, axis4, axis5) = plt.subplots(1, 3)

plot_system_dofs(axis1, system, "Sketch in Qestion a)")
plot_system_deflection(axis2, system, "Sketch in Question f)")

fig2.suptitle("Sketches in Question b)")

plot_element_fbd(axis3, system, element_1, arrow_scale=0.125)
plot_element_fbd(axis4, system, element_2, arrow_scale=0.125)
plot_element_fbd(axis5, system, element_3, arrow_scale=0.125)

print("Answer to Question c)")
for i, element in enumerate(system.elements):
    print(f"Element {i + 1} assembly matrix")
    print_matrix_rounded(element.assembly_matrix)
print("")

print("Answer to Question d)")
print("Local equivalent forcing for element 1")
print_matrix(element_1.get_local_equivalent_distributed_load())
print("Local equivalent forcing for element 2")
print_matrix(element_2.get_local_equivalent_distributed_load())
print("Overall forcing vector Q")
print_matrix(system.nodal_forces)
print("")

print("Answer to Question e) q = ")
print_matrix(system.dof_deflections, scale_coef=1e-3)
print("")

print("Answer to Question g)")
print("Support A:")
print_matrix(node_a.reaction_load, scale_coef=1e3)
print("Support D:")
print_matrix(node_d.reaction_load, scale_coef=1e3)
print("")

print("Answer to Question h)")
print("Element 3 displacement vector")
print_matrix(element_3.get_local_deflections())
print("Eement 3 Normal Strain:")
print(f"e = {element_3.get_axial_strain() * 1e6:.3f}ue")
print("Compression")

print("Answer to question i), Element 3 global deflection vector at x = L/2")
print_matrix(element_3.deflection_vector(0.5), scale_coef=1e-3)
print("")

plt.show()
