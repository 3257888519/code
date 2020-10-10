import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(t)

def x(n):
    return np.sin(n)


fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(221)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['bottom'].set_position(('data', 0))
ax1.yaxis.set_ticks_position('left')
ax1.spines['left'].set_position(('data',0))
t=np.arange(-3,3,0.01)
plt.plot(t,f(t),'r-')
plt.xlabel('t')
plt.ylabel('e**t')


ax2 = fig.add_subplot(222)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_ticks_position('bottom')
ax2.spines['bottom'].set_position(('data', 0))
ax2.yaxis.set_ticks_position('left')
ax2.spines['left'].set_position(('data',0))
n=np.arange(-9,9,0.01)
plt.plot(n,x(n),'r-')
plt.xlabel('t')
plt.ylabel('sin(t)')


ax3 = fig.add_subplot(223)
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
ax3.xaxis.set_ticks_position('bottom')
ax3.spines['bottom'].set_position(('data', 0))
ax3.yaxis.set_ticks_position('left')
ax3.spines['left'].set_position(('data',0))
for i in range(6):
    t1=[i,i]
    t2=[0,i+1]
    plt.plot(t1,t2,'b-')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.show()