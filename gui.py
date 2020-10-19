import tkinter
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist

def hanshu():
    fig = plt.figure(figsize=(4, 3),dpi=80)
    ax1= axisartist.Subplot(fig, 111)
    fig.add_axes(ax1)
    ax1.axis[:].set_visible(False)
    ax1.axis["x"] = ax1.new_floating_axis(0,0)
    ax1.axis["x"].set_axisline_style("->", size = 1.0)
    ax1.axis["x"].set_axis_direction("top")
    plt.xlim(-3,3)
    for i1 in np.arange(-3,3,1):
        if i1<0:
             t1=[i1,i1]
             t2=[0,0]
             plt.plot(t1,t2,'b*')
        else:
             t1=[i1,i1]
             t2=[0,1]
             plt.plot(t1,t2,'b*-')
    plt.title('x[n]=1,n>=0;0,n<0;')
    plt.show()

def hanshu1():
    fig = plt.figure(figsize=(4,3),dpi=80)
    ax2= axisartist.Subplot(fig, 111)
    fig.add_axes(ax2)
    ax2.axis[:].set_visible(False)
    ax2.axis["x"] = ax2.new_floating_axis(0,0)
    ax2.axis["x"].set_axisline_style("->", size = 1.0)
    ax2.axis["y"] = ax2.new_floating_axis(1,0)
    ax2.axis["y"].set_axisline_style("-|>", size = 1.0)
    ax2.axis["x"].set_axis_direction("top")
    ax2.axis["y"].set_axis_direction("right")
    plt.xlim(-5,5)
    plt.ylim(-0,2.0)
    for i2 in np.arange(-5,5):
        if i2!=0:
             t3=[i2]
             t4=[0]
             plt.plot(t3,t4,'b*')
        else:
            t3=[i2,i2]
            t4=[0,1]
            plt.plot(t3,t4,'b*-')
            plt.title('δ[n]')
            plt.xlabel('n')
    plt.show()

def hanshu2():
    fig = plt.figure(figsize=(4, 3),dpi=80)
    ax3= axisartist.Subplot(fig, 111)
    fig.add_axes(ax3)
    ax3.axis[:].set_visible(False)
    ax3.axis["x"] = ax3.new_floating_axis(0,0)
    ax3.axis["x"].set_axisline_style("->", size = 1.0)
    ax3.axis["y"] = ax3.new_floating_axis(1,0)
    ax3.axis["y"].set_axisline_style("-|>", size = 1.0)
    ax3.axis["x"].set_axis_direction("top")
    ax3.axis["y"].set_axis_direction("right")
    plt.xlim(-5,5)
    plt.ylim(-0, 1.5)
    for i3 in np.arange(-5,5):
           if i3<0:
                 t5=[i3]
                 t6=[0]
                 plt.plot(t5,t6,'b*')
           else:
                 t5=[i3,i3]
                 t6=[0,1]
                 plt.plot(t5,t6,'b*-')
    plt.title('u[n]')
    plt.xlabel('n')
    plt.show()

top=tkinter.Tk(className='hello world') 
top.geometry('400x200')
button = tkinter.Button(top)
button['text'] = 'x[n]'
button['command'] = hanshu
button.pack()


button = tkinter.Button(top)
button['text'] = 'δ[n]'
button['command'] = hanshu1
button.pack()

button = tkinter.Button(top)
button['text'] = 'u[n]'
button['command'] = hanshu2
button.pack()
top.mainloop()