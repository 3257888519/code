import tkinter
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
import pandas as pd

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

def hanshu3():
    fig = plt.figure(figsize=(4, 3),dpi=80)
    ax4= axisartist.Subplot(fig, 111)
    fig.add_axes(ax4)
    ax4.axis[:].set_visible(False)
    ax4.axis["x"] = ax4.new_floating_axis(0,0)
    ax4.axis["x"].set_axisline_style("->", size = 1.0)
    ax4.axis["y"] = ax4.new_floating_axis(1,0)
    ax4.axis["y"].set_axisline_style("-|>", size = 1.0)
    ax4.axis["x"].set_axis_direction("top")
    ax4.axis["y"].set_axis_direction("right")
    plt.xlim(-1,8)
    plt.ylim(0, 1.5)
    for i4 in np.arange(-1,8,1):
        if 0<=i4<=6-1:
             t7=[i4,i4]
             t8=[0,1]
             plt.plot(t7,t8,'r*-')
        else:
             t7=[i4]
             t8=[0]
             plt.plot(t7,t8,'r*')
    plt.title('$G_N[n]=1,0<=n<=6-1;0,n>=6$')
    plt.xlabel('n')
    plt.show()

def hanshu4():
    fig = plt.figure(figsize=(4, 3),dpi=80)
    ax5= axisartist.Subplot(fig, 111)
    fig.add_axes(ax5)
    ax5.axis[:].set_visible(False)
    ax5.axis["x"] = ax5.new_floating_axis(0,0)
    ax5.axis["x"].set_axisline_style("->", size = 1.0)
    ax5.axis["y"] = ax5.new_floating_axis(1,0)
    ax5.axis["y"].set_axisline_style("-|>", size = 1.0)
    ax5.axis["x"].set_axis_direction("top")
    ax5.axis["y"].set_axis_direction("right")
    plt.xlim(-0.2,5)
    plt.ylim(-0.2, 5)
    for i5 in np.arange(0,5,1):
        t9=[i5,i5]
        t10=[0,i5]
        plt.plot(t9,t10,'r*-')
    plt.title('$δ_N[n]$')
    plt.xlabel('n')
    plt.show()

def hanshu5():
    fig = plt.figure(figsize=(10,10),dpi=80)
    ax6= axisartist.Subplot(fig, 111)
    fig.add_axes(ax6)
    ax6.axis[:].set_visible(False)
    ax6.axis["x"] = ax6.new_floating_axis(0,0)
    ax6.axis["x"].set_axisline_style("->", size = 1.0)
    ax6.axis["y"] = ax6.new_floating_axis(1,0)
    ax6.axis["y"].set_axisline_style("-|>", size = 1.0)
    ax6.axis["x"].set_axis_direction("top")
    ax6.axis["y"].set_axis_direction("right")
    plt.xlim(-4.2,4.2)
    plt.ylim(-0.2, 1.2)
    for i6 in np.arange(-4,4):
         t11=[i6,i6]
         t12=[0,1]
         plt.plot(t11,t12,'r*-')
    plt.title('$δ_N[n]$')
    plt.xlabel('n')
    plt.show()

def hanshu6():
    fig = plt.figure(figsize=(4, 3),dpi=80)
    ax7= axisartist.Subplot(fig, 111)
    fig.add_axes(ax7)
    ax7.axis[:].set_visible(False)
    ax7.axis["x"] = ax7.new_floating_axis(0,0)
    ax7.axis["x"].set_axisline_style("->", size = 1.0)
    ax7.axis["y"] = ax7.new_floating_axis(1,0)
    ax7.axis["y"].set_axisline_style("-|>", size = 1.0)
    ax7.axis["x"].set_axis_direction("top")
    ax7.axis["y"].set_axis_direction("right")
    plt.xlim(-3.2,5.2)
    plt.ylim(-32, 32)
    a=2.0
    for i7 in np.arange(-3,5):
        t14=a**i7
        t12=[i7,i7]
        t13=[0,t14]
        plt.plot(t12,t13,'r*-')
    plt.title('$x[n]=ca^n$')
    plt.show()


top=tkinter.Tk(className='gui') 
top.geometry('200x300')
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

button = tkinter.Button(top)
button['text'] = 'G_N[n]'
button['command'] = hanshu3
button.pack()

button = tkinter.Button(top)
button['text'] = 'δN[n]'
button['command'] = hanshu4
button.pack()

button = tkinter.Button(top)
button['text'] = 'δN[n]'
button['command'] = hanshu5
button.pack()

button = tkinter.Button(top)
button['text'] = 'x[n]=ca^n'
button['command'] = hanshu6
button.pack()
top.mainloop()