from frame_solver import FrameSystem
from visualization import plot_system
import matplotlib.pyplot as plt

system = FrameSystem()

node_a = system.create_node("A", 0, 0)
node_a.fixed_joint()

node_b = system.create_node("B", 5, 0)

node_c = system.create_node("C", 7.5, 0)
node_c.force_y(-150e3)

node_d = system.create_node("D", 5, -2)
node_d.pin_joint()

left = system.create_element(node_a, node_b, 1, 640e-6, 200e9)
right = system.create_element(node_b, node_c, 1, 640e-6, 200e9)
system.weld_elements(left, right)

support = system.create_element(node_b, node_d, 400e-6, 1, 70e9)

system.solve()

print(support.get_local_deflections())

fig, axes = plt.subplots(1, 1)

plot_system(axes, system)

fig.legend()

plt.show()
