import numpy as np
import matplotlib.pyplot as plt

# initial condition
def phi_0(x):
    return np.array([
        np.sin(np.pi * ((xi - 40)/30))**2 if 40 <= xi < 70 else 0.0
        for xi in x
    ])

def upstream_scheme(x0, xmax, t0, tmax, u, dx, dt, tp):

    x = np.arange(x0, xmax + dx, dx)
    c = u*dt/dx
    phi = phi_0(x)

    plt.figure(figsize=(10,5))
    plt.plot(x, phi, label='t = {}'.format(int(t0)))
    plt.grid(True)
    
    t = t0
    while t<=tmax:
        phi_new = np.zeros_like(phi)

        if u > 0:
            phi_new[1:] = (1 - c) * phi[1:] + c * phi[:-1]
            
            # Periodic BC: phi(x=0) = phi(x=L), 
            # to compute phi_new(0), we need phi(-1) = phi(N-1) considering periodic bcs.

            phi_new[0]  = (1 - c) * phi[0]  + c * phi[-1]

        else:  # (for u<0 )

            phi_new[:-1] = (1 + c) * phi[:-1] - c * phi[1:]
            phi_new[-1]  = (1 + c) * phi[-1]  - c * phi[1]

        # update phi 
        phi = phi_new
        t += dt
    
        if np.mod(t, tp) < dt: 
            plt.plot(x, phi, label ='t = {}'.format(int(t)))
            plt.legend()

    # plt.title('Upwind scheme of LAE: $\Delta x= 0.1$, $u = 0.087 $, $\Delta t = 1.1$')
    # plt.savefig('Plots/upwind_laeq.png')
    plt.show()

u = 0.087
dx = 0.1
dt = dx/u
x0 = 0
xmax = 100
t0 = 0
tmax = 1000
tp = 200

upstream_scheme(x0, xmax, t0, tmax, u, dx, dt, tp)