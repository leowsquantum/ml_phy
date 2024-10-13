import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to generate the (2,1,1) state of hydrogen atom
def hydrogen211(n, max_r):
    def psi(x: tf.Tensor):
        r = tf.expand_dims(x[:, 0], axis=-1)
        theta = tf.expand_dims(x[:, 1], axis=-1)
        phi = tf.expand_dims(x[:, 2], axis=-1)
        return -r * tf.math.exp(-r / 2) * tf.math.sin(theta) * tf.math.cos(phi) / (8 * np.sqrt(np.pi))

    coordinates = tf.stack([
        max_r * tf.random.uniform((n,), dtype=tf.dtypes.float32, seed=0) ** (1. / 3),
        tf.random.uniform((n,), 0., 2 * np.pi, dtype=tf.dtypes.float32, seed=1),
        - tf.math.acos(tf.random.uniform((n,), -1., 1., dtype=tf.dtypes.float32, seed=2))
    ])

    x = coordinates[0] * tf.math.cos(coordinates[1]) * tf.math.sin(coordinates[2])
    y = coordinates[0] * tf.math.sin(coordinates[1]) * tf.math.sin(coordinates[2])
    z = coordinates[0] * tf.math.cos(coordinates[2])

    return x, y, z


# Generate hydrogen atom data
n = 1000
max_r = 5.0
x, y, z = hydrogen211(n, max_r)

# Convert tensors to numpy for plotting
x_np = x.numpy()
y_np = y.numpy()
z_np = z.numpy()

# 3D Scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_np, y_np, z_np, s=1, c=z_np, cmap='viridis', alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter plot of Hydrogen (2,1,1) State')

plt.show()