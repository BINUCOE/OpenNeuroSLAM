import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

exps = np.loadtxt("exp_map.txt")
odos = np.loadtxt("odo_map.txt")

fig = plt.figure(figsize=(11, 8), dpi=100)
ax1 = fig.add_subplot(121, projection='3d')
x1 = np.round(odos[:, 0], 2)
y1 = np.round(odos[:, 1], 2)
z1 = np.round(odos[:, 2], 2)
ax1.scatter(y1 * -0.8, x1 * 0.8, z1 * 0.1)
plt.title('Visual Odometry Map')

ax2 = fig.add_subplot(122, projection='3d')
x2 = np.round(exps[:, 0], 2)
y2 = np.round(exps[:, 1], 2)
z2 = np.round(exps[:, 2], 2)
ax2.scatter(y2 * 0.8, x2 * 0.8, z2 * 0.1)
plt.title('Multilayered Experience Map')

plt.tight_layout()
plt.tight_layout()
plt.show()
