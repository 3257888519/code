import numpy as np
import matplotlib.pyplot as plt

def x(t):
    return np.exp(t)*np.cos(8*t-200)

t=np.arange(-10,10,0.01)
plt.plot(t,x(t),'r-')
plt.show()