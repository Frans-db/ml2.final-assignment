import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D([0, 0], [0, 2], [1, 1])
ax.scatter3D([0], [1], [1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()