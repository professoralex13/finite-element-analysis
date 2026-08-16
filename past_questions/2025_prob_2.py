from frame_solver import FrameSystem
from visualization import plot_system_deflection, log_system_data
import matplotlib.pyplot as plt
import numpy as np

A = 1e-4
I = 3e-6
E = 200e9

system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_b = system.create_node("B", 0, 2)

node_c = system.create_node("C", 1.5, 2)

node_d = system.create_node("D", 2, 0)
node_d.fixed_joint()

element_1 = system.create_element(node_a, node_b, A, I, E)
element_1.add_distributed_shear_load(lambda t: -98.1e3 * (1 - t))

element_2 = system.create_element(node_b, node_c, A, I, E)
element_2.add_distributed_shear_load(lambda t: -10e3 * np.ones_like(t))

element_3 = system.create_element(node_a, node_c, A, I, E)
element_4 = system.create_element(node_c, node_d, A, I, E)

system.weld_elements(element_1, element_2)
system.weld_elements(element_1, element_3)
system.weld_elements(element_2, element_3)
system.weld_elements(element_2, element_4)
system.weld_elements(element_3, element_4)

system.solve()


fig, axes = plt.subplots(1, 1)

log_system_data(system)
plot_system_deflection(axes, system, 20)

plt.show()
