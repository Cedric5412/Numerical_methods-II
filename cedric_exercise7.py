import numpy as np
import matplotlib.pyplot as plt
import os

# Creates output folder if it doesn't exist
output_folder = './Plots/'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# initial condition
def phi_0(x):
    y = np.zeros(x.size)
    for i in range(x.size):
        if x[i] < 400 or x[i] > 600:
            y[i] = 0.0
        elif 400 <= x[i] < 500:
            y[i] = 0.1 * (x[i] - 400)
        elif 500 <= x[i] <= 600:
            y[i] = 20.0 - 0.1 * (x[i] - 400)
    return y 

def ftcs_diff(phi_now, c, dc):
    return phi_now - 0.5*c*(np.roll(phi_now, -1) - np.roll(phi_now, 1)) \
                + dc*(np.roll(phi_now, -1) - 2.0*phi_now + np.roll(phi_now, 1))

def ctcs_diff(phi_now, phi_old, c, dc):
    return phi_old - c*(np.roll(phi_now, -1) - np.roll(phi_now, 1)) \
                + 2.0*dc*(np.roll(phi_old, -1)  - 2.0*phi_old + np.roll(phi_old, 1))

def advect_diff(x0, xmax, t0, tmax, u, k, dx, dt, tp, alpha, beta):
    x = np.arange(x0, xmax+dx, dx)

    c = u*dt/dx
    dc = k*dt/(dx**2)
    
    phi_old = phi_0(x)
    phi_now = np.zeros_like(phi_old)

    phi_now = ftcs_diff(phi_old, c, dc)

    plt.figure(figsize=(14,8))
    plt.plot(x, phi_old, label='t = {}'.format(int(t0)))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x, t)$')
    plt.grid(True)
    
    t = t0 + dt
    while t<= tmax:
        phi_new = np.zeros_like(phi_now)

        phi_new = ctcs_diff(phi_now, phi_old, c, dc)

        # RAW filtering 
        d = alpha*(phi_old - 2*phi_now + phi_new)
        phi_now_smoothed = phi_now + beta*d
        phi_new_smoothed = phi_new + (1-beta)*d

        # update phi 
        phi_old = phi_now_smoothed
        phi_now = phi_new_smoothed
        t += dt
    
        if t%tp < dt: 
            plt.plot(x, phi_new, label ='t = {}'.format(int(t)))
            plt.legend(fontsize=12)

    plt.suptitle(r'$\dfrac{\partial \psi}{\partial t} + u \dfrac{\partial \psi}{\partial x} = K \dfrac{\partial^2 \psi}{\partial x^2}$', fontsize=12)
    plt.title(r'CTCS : $\Delta x= {}$, $u = {} $, $\Delta t$ = {}, $k={}$, $\alpha$ ={}, ' \
    r'$\beta$={}'.format(dx, u, round(dt, 2), k, alpha, beta), fontsize=12)

    output_filename = os.path.join(output_folder, 'advect_diffusion#{}_{}.png'.format(alpha, beta))
    plt.savefig(output_filename)

k = 0.029
u = 0.95
dx = 0.05
x0 = 0
xmax = 1000
t0 = 0
tmax = 2000
tp = 500

# Compute dt for stability
# Always have in mind a value of c. 
# Think of c = 0.4
dt_c = 0.4*dx/u
dt_dc = 0.4*dx**2/(2*k)

dt = np.min([dt_c, dt_dc])

advect_diff(x0, xmax, t0, tmax, u, k, dx, dt, tp, alpha=0.0, beta = 1.0)
advect_diff(x0, xmax, t0, tmax, u, k, dx, dt, tp, alpha=0.1, beta = 0.53)
