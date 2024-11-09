import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Create the time array
t = np.linspace(0, 1, 1000)

# Initial frequencies
f_x_init = 10.0  # Initial frequency for x(t)
f_y_init = 20.0  # Initial frequency for y(t)

# Compute initial x(t) and y(t)
x = np.cos(2 * np.pi * f_x_init * t)
y = np.sin(2 * np.pi * f_y_init * t)

# Create the figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.3)  # Make room for sliders

# Plot the initial 3D curve
[line] = ax.plot(x, y, t, lw=2)
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('t')
ax.set_title('3D Curve of Sinusoidal Functions')

# Set axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

# Create axes for the sliders
ax_freq_x = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_freq_y = plt.axes([0.1, 0.08, 0.8, 0.03])

# Create the sliders
freq_x_slider = Slider(
    ax=ax_freq_x,
    label='Frequency of x(t) [Hz]',
    valmin=0.,
    valmax=1000.,
    valinit=f_x_init,
    valstep=0.1,
)

freq_y_slider = Slider(
    ax=ax_freq_y,
    label='Frequency of y(t) [Hz]',
    valmin=0.,
    valmax=1000.,
    valinit=f_y_init,
    valstep=0.1,
)

# Define the update function
def update(val):
    f_x = freq_x_slider.val
    f_y = freq_y_slider.val
    # Recalculate x(t) and y(t) with new frequencies
    x = np.cos(2 * np.pi * f_x * t)
    y = np.sin(2 * np.pi * f_y * t)
    # Update the data of the plotted line
    line.set_data(x, y)
    line.set_3d_properties(t)
    fig.canvas.draw_idle()

# Connect the sliders to the update function
freq_x_slider.on_changed(update)
freq_y_slider.on_changed(update)

# Display the plot
plt.show()
