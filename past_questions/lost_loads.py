import matplotlib.pyplot as plt
import numpy as np
from frame_solver import FrameSystem
from visualization import plot_system_deflection, log_system_data

A = 4e-4
I = 1e-5
E = 200e9

system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_b = system.create_node("B", 3, 0)

element_1 = system.create_element(node_a, node_b, A, I, E)
element_1.add_distributed_shear_load(lambda t: -10e3 * np.ones_like(t))

system.solve()

fig, axes = plt.subplots(1, 1)

log_system_data(system)
plot_system_deflection(axes, system, deflection_scaling=20)

plt.show()
