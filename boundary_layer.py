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
        y[i] = 293.0
    return y

def diffusion_ftcs_scheme(x0, xmax, t0, tmax, k, dx, dt, tp, Q, temp0):

    x = np.arange(x0, xmax + dx, dx)
    n = x.size
    L = xmax - x0
    
    phi_now = phi_0(x)

    plt.figure(figsize=(12,8))
    plt.plot(phi_now, x, label=f't = {int(t0%3600)}')
    plt.ylim(x0, xmax)
    plt.ylabel(r'$z\;(m)$')
    plt.xlabel(r'$T\;(K)$')
    # plt.grid(True)
    
    t = t0 + dt
    while t <= tmax:
        phi_new = np.zeros_like(phi_now)

        # BCS
        phi_new[0] = temp0

        for i in range(1, n-1):
            phi_new[i] = phi_now[i] + ((k*dt)/dx**2 )* (phi_now[i+1] - 2*phi_now[i] + phi_now[i-1]) + Q*dt
        
        phi_new[-1] = phi_new[-2]
        phi_now[:] = phi_new[:]
        t += dt
        if abs(t % tp) < dt:
            plt.plot(phi_new, x, label=f'$t$ = {int(t/3600)} h')
            plt.legend()
    
    title = rf'Parameters: $k$ = {k}, $dz$={dx}, $dt$={round(dt, 2)}, $Q={Q*24*3600} \;K/day$'
    plt.suptitle(r'$\dfrac{\partial \phi}{\partial t} = K \dfrac{{\partial}^2 \phi}{\partial z^2} + Q $')
    plt.title(title)
    fname = os.path.join(output_folder, 'PBL_diffusion.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
    
Q = -1.5/(24*3600)
temp0 = 293.0
x0=0
xmax=1000
t0=0
tmax=6*3600
k=1
dx=1.0
tp=2*3600
dt=0.5*(dx**2/(2*k))

diffusion_ftcs_scheme(x0, xmax, t0, tmax, k, dx, dt, tp, Q, temp0)