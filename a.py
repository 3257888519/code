import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

def x(t):
    return np.exp(-t)*np.cos(8*t-200)


fig = plt.figure(figsize=(10, 8),dpi=80)
ax1= axisartist.Subplot(fig, 111)
fig.add_axes(ax1)
ax1.axis[:].set_visible(False)
ax1.axis["x"] = ax1.new_floating_axis(0,0)
ax1.axis["x"].set_axisline_style("->", size = 1.0)
ax1.axis["y"] = ax1.new_floating_axis(1,0)
ax1.axis["y"].set_axisline_style("-|>", size = 1.0)
ax1.axis["x"].set_axis_direction("top")
ax1.axis["y"].set_axis_direction("right")
plt.xlim(-5,5)
plt.ylim(-100,100)
t=np.arange(-5,5,0.01)
plt.plot(t,x(t),'r-')
plt.title('x(t)=cos(ωt+θ)*e**(σt)')
plt.xlabel('t')
plt.show()