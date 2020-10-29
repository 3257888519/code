# 第二次实验

import numpy as np

import matplotlib.pyplot as plt

import mpl_toolkits.axisartist as axisartist



def f(t):

  return np.exp(t)



def x(n):

  return np.sin(n)





fig = plt.figure(figsize=(10, 8),dpi=80)

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

plt.title('e**t')





ax2= axisartist.Subplot(fig, 332)

fig.add_axes(ax2)

ax2.axis[:].set_visible(False)

ax2.axis["x"] = ax2.new_floating_axis(0,0)

ax2.axis["x"].set_axisline_style("->", size = 1.0)

ax2.axis["y"] = ax2.new_floating_axis(1,0)

ax2.axis["y"].set_axisline_style("-|>", size = 1.0)

ax2.axis["x"].set_axis_direction("top")

ax2.axis["y"].set_axis_direction("right")

plt.xlim(-9,9)

plt.ylim(-1, 1)

n=np.arange(-9,9,0.01)

plt.plot(n,x(n),'r-')

plt.xlabel('t')

plt.title('sin(t)')





ax3= axisartist.Subplot(fig, 333)

fig.add_axes(ax3)

ax3.axis[:].set_visible(False)

ax3.axis["x"] = ax3.new_floating_axis(0,0)

ax3.axis["x"].set_axisline_style("->", size = 1.0)

ax3.axis["y"] = ax3.new_floating_axis(1,0)

ax3.axis["y"].set_axisline_style("-|>", size = 1.0)

ax3.axis["x"].set_axis_direction("top")

ax3.axis["y"].set_axis_direction("right")

plt.xlim(-5,5)

plt.ylim(-1, 2)

t1=[0,1,1]

t2=[1,1,0]

plt.xlabel('t')

plt.title('energy signal')

plt.plot(t1,t2,'b-')



ax4= axisartist.Subplot(fig, 334)

fig.add_axes(ax4)

ax4.axis[:].set_visible(False)

ax4.axis["x"] = ax4.new_floating_axis(0,0)

ax4.axis["x"].set_axisline_style("->", size = 1.0)

ax4.axis["y"] = ax4.new_floating_axis(1,0)

ax4.axis["y"].set_axisline_style("-|>", size = 1.0)

ax4.axis["x"].set_axis_direction("top")

ax4.axis["y"].set_axis_direction("right")

plt.xlim(-10,12)

plt.ylim(-0.2, 1)

x1=[0,5,13]

y1=[0,1,1]

plt.plot(x1,y1,'b-')

plt.xlabel('t')

plt.title('power signal')



def u(t):

  return np.array(t>0,dtype=np.int)



ax5= axisartist.Subplot(fig, 335)

fig.add_axes(ax5)

ax5.axis[:].set_visible(False)

ax5.axis["x"] = ax5.new_floating_axis(0,0)

ax5.axis["x"].set_axisline_style("->", size = 1.0)

ax5.axis["y"] = ax5.new_floating_axis(1,0)

ax5.axis["y"].set_axisline_style("-|>", size = 1.0)

ax5.axis["x"].set_axis_direction("top")

ax5.axis["y"].set_axis_direction("right")

plt.xlim(-12,12)

plt.ylim(-0.2, 2)

t3=np.arange(-10,10,0.01)

plt.plot(t3,u(t3),'r-')

plt.xlabel('t')

plt.title('unit step signal')



ax6= axisartist.Subplot(fig, 336)

fig.add_axes(ax6)

ax6.axis[:].set_visible(False)

ax6.axis["x"] = ax6.new_floating_axis(0,0)

ax6.axis["x"].set_axisline_style("->", size = 1.0)

ax6.axis["y"] = ax6.new_floating_axis(1,0)

ax6.axis["y"].set_axisline_style("-|>", size = 1.0)

ax6.axis["x"].set_axis_direction("top")

ax6.axis["y"].set_axis_direction("right")

plt.xlim(0,4)

plt.ylim(-0.2, 1.2)

t4=np.arange(0,4,0.01)

plt.plot(t4,u(t4-1)-u(t4-3),'r-')

plt.xlabel('t')

plt.title('u(t-1)-u(t-3)')



def g(t):

  return np.sin(t)/t



ax7= axisartist.Subplot(fig, 337)

fig.add_axes(ax7)

ax7.axis[:].set_visible(False)

ax7.axis["x"] = ax7.new_floating_axis(0,0)

ax7.axis["x"].set_axisline_style("->", size = 1.0)

ax7.axis["y"] = ax7.new_floating_axis(1,0)

ax7.axis["y"].set_axisline_style("-|>", size = 1.0)

ax7.axis["x"].set_axis_direction("top")

ax7.axis["y"].set_axis_direction("right")

plt.xlim(-15,15)

plt.ylim(-0.4, 1.0)

t5=np.arange(-15,15,0.01)

plt.plot(t5,g(t5),'r-')

plt.xlabel('t')

plt.title('Sa(t)')



ax8= axisartist.Subplot(fig, 338)

fig.add_axes(ax8)

ax8.axis[:].set_visible(False)

ax8.axis["x"] = ax8.new_floating_axis(0,0)

ax8.axis["x"].set_axisline_style("->", size = 1.0)

ax8.axis["y"] = ax8.new_floating_axis(1,0)

ax8.axis["y"].set_axisline_style("-|>", size = 1.0)

ax8.axis["x"].set_axis_direction("top")

ax8.axis["y"].set_axis_direction("right")

plt.xlim(-1.0,1.0)

plt.ylim(-0.2, 1.2)

t6=[0,0]

t7=[0,1]

plt.plot(t6,t7,'r-')

plt.title("impulse signal")

plt.xlabel('t')

plt.show( )

# 第二次实验补充

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

plt.show( )
