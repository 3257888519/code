import numpy as np
import matplotlib.pyplot as plt
 
x=np.linspace(0,10,1000)
y=np.sin(x)
z=np.cos(x**2)
plt.figure(figsize=(8,4))
 
plt.plot(x,y,label='$sin(x)$',color='red',linewidth=3)
 
plt.plot(x,z,'g--',label='$cos(x^2)$',lw=3)
 
plt.xlabel('Time(s)')
plt.ylabel('volt')
plt.title('First python firgure')
plt.ylim(-1.2,1.2)
plt.legend()
 
plt.show()