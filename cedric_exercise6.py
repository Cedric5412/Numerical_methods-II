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


def thomas_diffusion(x0, xmax, t0, tmax, k, dx, dt, tp, temp0):

    x = np.arange(x0, xmax + dx, dx)
    n = x.size
   
    alpha = k*dt/(dx**2)
    phi_now = phi_0(x)
    phi_new = np.zeros_like(phi_now)

    plt.figure(figsize=(12,8))
    plt.plot(x, phi_now, label=f't = {int(t0%3600)}')
    plt.xlim(0, 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x, t)$')
    plt.grid(True)
    
    t = t0 + dt

    while t<=tmax:
        A = np.zeros(n-2)
        B = np.zeros(n-1)
        C = np.zeros(n-2)
        for i in range(n-2):
            A[i] = -alpha
            C[i] = -alpha
            B[i] = 1 + 2.0*alpha
        for i in range(n-1):
            B[i] = 1 + 2.0*alpha
        F = np.zeros(n-1)
        delta = np.zeros_like(F)

        F[0]= 0
        delta[0] = temp0

        phi_new = np.zeros_like(phi_now)

        for j in range(n-2):
            F[j+1] = C[j] / (B[j+1] - A[j] * F[j])
            delta[j+1] = (phi_now[j+1] - A[j] * delta[j]) / (B[j+1] - A[j] * F[j])
                
        phi_new[-1] = temp0

        for j in reversed(range(n-1)):
            phi_new[j] = delta[j] - F[j] * phi_new[j+1]

        phi_now = phi_new
        t = t + dt

        if abs(t % tp) < dt:
            plt.plot(x, phi_new, label=f'$t$ = {int(t/3600)} h')
            plt.legend()
        
    title = rf'Parameters: $k$ = {k}, $dx$={dx}, $dt$={round(dt, 2)}'
    plt.suptitle(r'Implicit scheme: $\dfrac{\partial \phi}{\partial t} = K \dfrac{{\partial}^2 \phi}{\partial x^2}$')
    plt.title(title)
    fname = os.path.join(output_folder, 'Heat_implicit.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

temp0 = 273.15
x0 = 0
xmax = 1
t0 = 0
tmax = 21600
k = 2.9e-5
dx = 0.01
tp = 3600
dt = 360.0

thomas_diffusion(x0, xmax, t0, tmax, k, dx, dt, tp, temp0)