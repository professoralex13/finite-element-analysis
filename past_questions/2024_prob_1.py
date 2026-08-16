import numpy as np
import matplotlib.pyplot as plt
from frame_solver import FrameSystem
from visualization import plot_system_deflection, log_system_data

A = 3e-4
I = 5e-6
E = 200e9


system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_b = system.create_node("B", 5, 0)
node_b.force_x(15e3)
node_b.force_y(-20e3)

node_c = system.create_node("C", 3.2, -2.4)
node_c.x_slider_joint()

element_1 = system.create_element(node_a, node_b, A, I, E)
element_1.add_distributed_shear_load(lambda t: -30e3 * np.ones_like(t))

element_2 = system.create_element(node_a, node_c, A, I, E)

element_3 = system.create_element(node_c, node_b, A, I, E)

system.solve()

fig, axes = plt.subplots(1, 1)

log_system_data(system)
plot_system_deflection(axes, system, deflection_scaling=5)

plt.show()
