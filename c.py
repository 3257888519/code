import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(10)
y=np.sin(x)

plt.figure(figsize=(10,8))

plt.plot(x,y,label=r'$\alpha_v$')  #在两个$号之间输出
plt.title(r'$\alpha$')
plt.legend(fontsize=16)
plt.xlabel(r'$\G_N[n]$',fontsize=16)  #下标用_来表示，如果下标或者上标有多个字符，用花括号括起来
#plt.xlabel(r'$T$',fontsize=16)  #输出斜体T
plt.show()