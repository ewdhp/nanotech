"""
ac_dc_demo.py

A simple script to demonstrate the difference between AC (Alternating Current) and DC (Direct Current).
Plots both waveforms for visualization.
"""
import numpy as np
import matplotlib.pyplot as plt

# Time array
T = 1  # period for AC
f = 1  # frequency (Hz)
t = np.linspace(0, 2 * T, 1000)

# DC: constant voltage
V_dc = 5  # volts
V_dc_array = V_dc * np.ones_like(t)

# AC: sinusoidal voltage
V_ac = 5  # peak volts
V_ac_array = V_ac * np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 5))
plt.plot(t, V_dc_array, label='DC (Direct Current)', color='red')
plt.plot(t, V_ac_array, label='AC (Alternating Current)', color='blue')
plt.title('AC vs DC Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
