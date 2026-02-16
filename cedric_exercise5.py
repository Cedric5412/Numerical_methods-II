import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Creates output folder if it doesn't exist
output_folder = './Plots/'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# Initial condition
def phi_0(x):
    y = np.zeros(x.size)
    for i in range(x.size):
        if 0.0<= x[i] <= 0.5:
            y[i] = 273.15 + 20*x[i] + np.sin(50*x[i]*np.pi)
        else :
            y[i] = 273.15 + 20 - 20*x[i] + np.sin(50*x[i]*np.pi)
    return y

def diffusion_ftcs_scheme(x0, xmax, t0, tmax, k, dx, dt, tp):

    x = np.arange(x0, xmax + dx, dx)
    n = x.size
    L = xmax - x0
    
    phi_now = phi_0(x)


    plt.figure(figsize=(12,8))
    plt.plot(x, phi_now, label=f't = {int(t0%3600)}')
    plt.xlim(0, 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x, t)$')
    plt.grid(True)
    
    t = t0 + dt
    while t <= tmax:
        phi_new = np.zeros_like(phi_now)

        # BCS
        phi_new[0] = 273.15
        phi_new[-1] = 273.15

        for i in range(1, n-1):
            phi_new[i] = ((k*dt)/dx**2 )* (phi_now[i+1] - 2*phi_now[i] + phi_now[i-1]) + phi_now[i]
        phi_now[:] = phi_new[:]
        t += dt
        if abs(t % tp) < dt:
            plt.plot(x, phi_now, label=f'$t$ = {int(t/3600)} h')
            plt.legend()
    
    title = rf'Parameters: $k$ = {k}, $dx$={dx}, $dt$={round(dt, 2)}'
    plt.suptitle(r'$\dfrac{\partial \phi}{\partial t} = K \dfrac{{\partial}^2 \phi}{\partial x^2}$')
    plt.title(title)
    fname = os.path.join(output_folder, 'Heat.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

x0=0
xmax=1
t0=0
tmax=21600
k=2.9e-5
dx=0.01
tp=3600
dt=0.5*(dx**2/(2*k))

diffusion_ftcs_scheme(x0, xmax, t0, tmax, k, dx, dt, tp)