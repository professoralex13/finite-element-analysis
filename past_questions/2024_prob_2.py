from frame_solver import FrameSystem
from visualization import plot_system_deflection, log_system_data, print_matrix
import matplotlib.pyplot as plt
import numpy as np

A = 3e-4
I = 5e-6
E = 200e9


system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_b = system.create_node("B", 0, 2.5)

node_c = system.create_node("C", 3, 2)
node_c.fix_x()
node_c.fix_rotation()

element_1 = system.create_element(node_a, node_b, A, I, E)

element_2 = system.create_element(node_b, node_c, A, I, E)

w = -4e3

element_2.add_distributed_shear_load(
    lambda t: np.ones_like(t) * w * np.cos(element_2.alpha())
)

element_2.add_distributed_axial_load(
    lambda t: np.ones_like(t) * w * np.sin(element_2.alpha())
)

system.weld_elements(element_1, element_2)

system.solve()

fig, axes = plt.subplots(1, 1)

log_system_data(system)
plot_system_deflection(axes, system, 10)

plt.show()
