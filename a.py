import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

def f(t):
    return np.exp(t)

def x(n):
    return np.sin(n)


fig = plt.figure(figsize=(10, 10),dpi=80)
ax1= axisartist.Subplot(fig, 331)
fig.add_axes(ax1)
ax1.axis[:].set_visible(False)
ax1.axis["x"] = ax1.new_floating_axis(0,0)
ax1.axis["x"].set_axisline_style("->", size = 1.0)
ax1.axis["y"] = ax1.new_floating_axis(1,0)
ax1.axis["y"].set_axisline_style("-|>", size = 1.0)
ax1.axis["x"].set_axis_direction("top")
ax1.axis["y"].set_axis_direction("right")
plt.xlim(-3,5)
plt.ylim(-5,25)
t=np.arange(-3,3,0.01)
plt.plot(t,f(t),'r-')
plt.plot(t,f(-t),'r-')
plt.plot(t,f(0*t),'r-')
plt.xlabel('t')
plt.ylabel('e**t')
plt.show()