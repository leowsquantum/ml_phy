import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Create the figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Adjust the layout to make room for the sliders
plt.subplots_adjust(left=0.1, bottom=0.3)

# Define the initial frequencies
omega1_init = 4.0  # Initial frequency ω₁
omega2_init = 4.0  # Initial frequency ω₂

# Create a grid of x and y values
x = np.linspace(-np.pi, np.pi, 200)
y = np.linspace(-np.pi, np.pi, 200)
X, Y = np.meshgrid(x, y)

# Compute the initial z values
Z = np.cos(omega1_init * X) * np.sin(omega2_init * Y)

# Plot the initial surface
surface = ax.plot_surface(X, Y, Z, cmap='viridis')

# Set the labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot of z = cos(ω₁ x) × sin(ω₂ y)')

# Set the z-axis limits
ax.set_zlim(-1, 1)

# Create axes for the sliders
ax_omega1 = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_omega2 = plt.axes([0.1, 0.08, 0.8, 0.03])

# Create the sliders
omega1_slider = Slider(
    ax=ax_omega1,
    label='Frequency ω₁',
    valmin=0,
    valmax=30,
    valinit=omega1_init,
    valstep=0.1,
)

omega2_slider = Slider(
    ax=ax_omega2,
    label='Frequency ω₂',
    valmin=0,
    valmax=30,
    valinit=omega2_init,
    valstep=0.1,
)

# Define the update function
def update(val):
    # Get the current slider values
    omega1 = omega1_slider.val
    omega2 = omega2_slider.val
    # Recalculate Z with the new frequencies
    Z = np.cos(omega1 * X) * np.sin(omega2 * Y)
    # Clear the current surface
    ax.clear()
    # Re-plot the surface
    surface = ax.plot_surface(X, Y, Z, cmap='viridis')
    # Set the labels and title again since ax.clear() removes them
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Surface Plot of z = cos(ω₁ x) × sin(ω₂ y)')
    # Set the z-axis limits
    ax.set_zlim(-1, 1)
    # Redraw the figure canvas
    fig.canvas.draw_idle()

# Connect the sliders to the update function
omega1_slider.on_changed(update)
omega2_slider.on_changed(update)

# Display the plot
plt.show()
