import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return - y/2 + 4*np.exp(-x/2) * np.cos(4*x)

# exact solution
def yy(x):
    return np.exp(-x/2) * np.sin(4*x)

def euler_method(x0, y0, xf, dx):

    n = int(xf/dx)
    xlist = np.zeros(n+1)
    ylist = np.zeros(n+1)
    elist = np.zeros_like(ylist)

    xlist[0] = x0
    ylist[0] = y0
    elist[0] = y0 = yy(x0)

    for i in range(n):
        xlist[i+1] = xlist[i] + dx
        ylist[i+1] = ylist[i] + f(xlist[i], ylist[i]) * dx

        elist[i+1] = ylist[i+1] - yy(xlist[i+1])

    return xlist, elist

def runge_kutta(x0, y0, xf, dx):

    n = int(xf/dx)
    xlist = np.zeros(n+1)
    ylist = np.zeros(n+1)
    elist = np.zeros_like(ylist)

    xlist[0] = x0
    ylist[0] = y0
    elist[0] = yy(x0)

    for i in range(n):
        xlist[i+1] = xlist[i] + dx

        # compute y*(i+1)
        ylist[i+1] = ylist[i] +  f(xlist[i], ylist[i]) * dx  

        # actual y(i+1)
        ylist[i+1] = ylist[i] + (dx/2) * ( f(xlist[i], ylist[i]) + f(xlist[i+1], ylist[i+1]))

        elist[i+1] = ylist[i+1] - yy(xlist[i+1])

    return xlist, elist




x0 = 0.0
y0 = 0.0
xf = 10.0
dx = 0.1

x, e = euler_method(x0, y0, xf, dx)
xrk, erk = runge_kutta(x0, y0, xf, dx)
# xx = np.linspace(0, 10, 1000)
# yy_xx = yy(xx)

plt.figure(figsize=(10,5))
plt.title('Error w.r.t analytical solution')
plt.plot(x, e, label='Euler')
# plt.plot(xx, yy_xx, label='Analytical')
plt.plot(xrk, erk, label='Runge-Kutta')
plt.legend()
plt.savefig('error_ode.png')
