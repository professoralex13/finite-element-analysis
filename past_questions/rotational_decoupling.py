import numpy as np
import matplotlib.pyplot as plt
from frame_solver import FrameSystem
from visualization import (
    plot_system_deflection,
    plot_system_dofs,
    log_system_data,
    print_matrix,
    print_matrix_rounded,
)

A = 1e-3
I = 1e-5
E = 200e9

system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.force_y(-100e3)

node_b = system.create_node("B", 4 / 3, 0)

node_c = system.create_node("C", 8 / 3, 0)
node_c.pin_joint()

node_d = system.create_node("D", 8 / 3, -1)
node_d.pin_joint()


element_1 = system.create_element(node_a, node_b, A, I, E)
element_2 = system.create_element(node_b, node_c, A, I, E)
support = system.create_element(node_b, node_d, 5e-4, 5e-6, E)

node_b.add_extra_rotation_dof(0, [support])

system.solve()

fig, axis = plt.subplots(1, 1)

plot_system_dofs(axis, system, arrow_scale=0.1)

plt.show()
