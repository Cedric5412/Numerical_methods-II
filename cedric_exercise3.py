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
        if x[i] < 200 or x[i] > 300:
            y[i] = 0.1
        elif  200 <= x[i] < 250:
            y[i] = 2.0
        elif 250 <= x[i] <= 300:
            y[i] = 1.0    
    return y     

def upstream_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha, beta):
    x = np.arange(x0, xmax+dx, dx)
    # index

    N = x.size - 1
    c = u*dt/dx
    
    phi0 = phi_0(x)
    phi1 = np.zeros_like(phi0)
    if u < 0:
        phi1[:N] = (1 + c) * phi0[:N] - c * phi0[1:]
        phi1[N]  = (1 + c) * phi0[N]  - c * phi0[1]
    else :
        phi1[1:] = (1 - c) * phi0[1:] + c * phi0[:N]
        phi1[0]  = (1 - c) * phi0[0]  + c * phi0[N-1]

    plt.figure(figsize=(14,8))
    plt.plot(x, phi0, label='t = {}'.format(int(t0)))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x, t)$')
    plt.grid(True)
    
    t = t0 + dt
    while t<= tmax:
        phi_new = np.zeros_like(phi1)
        # BCS
        phi_new[0]  = phi0[0] - c*(phi1[1] - phi1[N-1])
        phi_new[N]  = phi0[N] - c*(phi1[1] - phi1[N-1])
        # CTCS
        phi_new[1:N] = phi0[1:N] - c*(phi1[2:] - phi1[:N-1])
        

        d = alpha*(phi0 - 2*phi1 + phi_new)
        phi1_smoothed = phi1 + beta*d
        phi_new_smoothed = phi_new + (1-beta)*d

        # update phi 
        phi0 = phi1_smoothed
        phi1 = phi_new_smoothed
        t += dt
    
        if t%tp < dt: 
            plt.plot(x, phi_new, label ='t = {}'.format(int(t)))
            plt.legend()

    plt.title(r'CTCS scheme for LAE: $\Delta x= {}$, $u = {} $, $\Delta t$ = {}, $\alpha$ ={}, ' \
    r'$\beta$={}'.format(dt, u, round(dt, 2), alpha, beta))

    output_filename = os.path.join(output_folder, 'ctcs_laeq{}_{}.png'.format(alpha, beta))
    plt.savefig(output_filename)

u = -0.31
dx = 0.1
dt = 0.08
x0 = 0
xmax = 500
t0 = 0
tmax = 1000
tp = 200

upstream_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha=0, beta=1)
upstream_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha=0.1, beta=1)
upstream_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha=0.05, beta=0.53)
