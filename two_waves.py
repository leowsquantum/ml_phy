import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the time range
t = np.linspace(0, 1, 1000)

# Initial frequencies of the sine waves
f1_init = 100  # Frequency of the first sine wave in Hz
f2_init = 200  # Frequency of the second sine wave in Hz

# Compute the initial superposed signal
signal = np.sin(2 * np.pi * f1_init * t) + np.sin(2 * np.pi * f2_init * t)

# Create the plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.3)  # Make room for sliders

# Plot the initial signal
[line] = ax.plot(t, signal, lw=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.set_title('Superposition of Two Sine Waves')
ax.set_xlim(0, 1)
ax.set_ylim(-2, 2)

# Create axes for the sliders
ax_freq1 = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_freq2 = plt.axes([0.1, 0.08, 0.8, 0.03])

# Create the sliders
freq1_slider = Slider(
    ax=ax_freq1,
    label='Frequency 1 [Hz]',
    valmin=1,
    valmax=1000,
    valinit=f1_init,
)

freq2_slider = Slider(
    ax=ax_freq2,
    label='Frequency 2 [Hz]',
    valmin=1,
    valmax=1000,
    valinit=f2_init,
)

# Define the update function
def update(val):
    f1 = freq1_slider.val
    f2 = freq2_slider.val
    # Recalculate the signal with new frequencies
    new_signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    line.set_ydata(new_signal)
    fig.canvas.draw_idle()

# Connect the sliders to the update function
freq1_slider.on_changed(update)
freq2_slider.on_changed(update)

# Display the plot
plt.show()
