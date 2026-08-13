from frame_solver import FrameSystem
from visualization import plot_system
import matplotlib.pyplot as plt
import numpy as np

system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_b = system.create_node("B", 0, 3)
node_b.force_x(10e3)

node_c = system.create_node("C", 4.5, 3)
node_c.force_x(10e3)

node_d = system.create_node("D", 4.5, 0)
node_d.fixed_joint()

I = 1e-5
A = 5e-4
E = 200e9

left = system.create_element(node_a, node_b, A, I, E)
top = system.create_element(node_b, node_c, A, I, E)
right = system.create_element(node_c, node_d, A, I, E)

system.weld_elements(left, top)
system.weld_elements(top, right)

top.add_distributed_load(lambda t: -10e3 * np.ones_like(t))
top.add_point_load(-50e3, 0.5)

system.solve()

fig, axes = plt.subplots(1, 1)

plot_system(axes, system)

fig.legend()

plt.show()
