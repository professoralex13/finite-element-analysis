import numpy as np
import matplotlib.pyplot as plt
from frame_solver import FrameSystem
from visualization import plot_system, log_system_data

A = 2e-4
I = 1e-4
E = 200e9

depth = 3

system = FrameSystem()

node_a = system.create_node(
    "A", 4 - 4 * np.cos(np.radians(30)), depth + 4 * np.sin(np.radians(30))
)
node_a.fixed_joint()

node_b = system.create_node("B", 4, depth)

node_c = system.create_node("C", 4, 0)
node_c.pin_joint()

node_d = system.create_node("D", 0, 0)
node_d.fixed_joint()

G = 9.81
width = 4
density = 1e3

max_pressure = density * G * depth


element_1 = system.create_element(node_a, node_b, A, I, E)
element_2 = system.create_element(node_b, node_c, A, I, E)
element_2.add_distributed_shear_load(lambda t: -max_pressure * width * t)
element_3 = system.create_element(node_d, node_c, A, I, E)
element_3.add_distributed_shear_load(lambda t: max_pressure * width * np.ones_like(t))

system.weld_elements(element_1, element_2)
system.weld_elements(element_2, element_3)

system.solve()

print(element_1.get_axial_force() * 1e-3)
print(element_1.get_axial_stress() * 1e-6)
print(element_1.get_axial_strain() * 1e6)

fig, axes = plt.subplots(1, 1)

log_system_data(system)
plot_system(axes, system, 50)

plt.show()
