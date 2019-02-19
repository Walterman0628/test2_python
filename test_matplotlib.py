#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
# test 1
x = np.linspace(-np.pi, np.pi, 256, endpoint = True)
c, s = np.cos(x), np.sin(x)
plt.figure(1)
plt.plot(x, c, color = 'blue', linewidth = 1.0, linestyle = '-', label = 'COS', alpha = 0.5)
plt.plot(x, s)
plt.title('COS AND SIN')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1, 0, +1], [r'$-1$', r'$0$', r'$+1$'])
t = 2*np.pi/3
plt.plot([t,t],[0,np.cos(t)], color ='blue', linewidth=2.5, linestyle="--")
plt.scatter([t,],[np.cos(t),], 50, color ='blue')
plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',xy=(t, np.sin(t)), xycoords='data',xytext=(+10, +30), textcoords='offset points', fontsize=16,arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.plot([t,t],[0,np.sin(t)], color ='red', linewidth=2.5, linestyle="--")
plt.scatter([t,],[np.sin(t),], 50, color ='red')
plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',xy=(t, np.cos(t)), xycoords='data',xytext=(-90, -50), textcoords='offset points', fontsize=16,arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.legend(loc='upper left')
plt.show()
# test 2
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y = np.sin(2*x)
plt.plot(x, y+1, color='blue', alpha=1.00)
plt.plot(x, y-1, color='red', alpha=1.00)
plt.show()
# test 3
x = np.random.normal(0, 1, 1024)
y = np.random.normal(0, 1, 1024)
plt.scatter(x, y)
plt.show()