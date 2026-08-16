import numpy as np
import matplotlib.pyplot as plt
from frame_solver import FrameSystem
from visualization import plot_system, log_system_data

A = 2e-4
I = 6e-6
E = 200e9


system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fix_x()
node_a.fix_rotation()

node_b = system.create_node("B", -3, 0)

node_c = system.create_node("C", -6, -4)
node_c.pin_joint()

node_d = system.create_node("D", 0, -4)
node_d.fix_x()
node_d.fix_rotation()

element_1 = system.create_element(node_b, node_a, A, I, E)
element_2 = system.create_element(node_c, node_b, A, I, E)
element_3 = system.create_element(node_b, node_d, A, I, E)
element_4 = system.create_element(node_c, node_d, A, I, E)
element_4.add_distributed_shear_load(lambda t: -40e3 * np.ones_like(t))

system.weld_elements(element_1, element_2)
system.weld_elements(element_1, element_3)
system.weld_elements(element_2, element_4)
system.weld_elements(element_3, element_4)

system.solve()


fig, axes = plt.subplots(1, 1)

log_system_data(system)
plot_system(axes, system, 5)

plt.show()
