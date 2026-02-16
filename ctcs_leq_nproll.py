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

def ftfs(phi_now, c):
    return (1+c)*phi_now - c * np.roll(phi_now, -1)

def ftbs(phi_now, c):
    return (1-c)*phi_now + c * np.roll(phi_now, 1)

def ctcs(phi_now, phi_old, c):
    return phi_old - c * (np.roll(phi_now, -1) - np.roll(phi_now, 1))

def ctcs_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha, beta):
    x = np.arange(x0, xmax+dx, dx)
    # index
    N = x.size - 1
    c = u*dt/dx
    
    phi0 = phi_0(x)
    phi1 = np.zeros_like(phi0)

    if u < 0:
        phi1 = ftfs(phi0, c)
    else :
        phi1 = ftbs(phi0, c)

    plt.figure(figsize=(14,8))
    plt.plot(x, phi0, label='t = {}'.format(int(t0)))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x, t)$')
    plt.grid(True)
    
    t = t0 + dt
    while t<= tmax:
        phi_new = np.zeros_like(phi1)
        phi_new = ctcs(phi1, phi0, c)

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
    r'$\beta$={}'.format(dx, u, round(dt, 2), alpha, beta))

    output_filename = os.path.join(output_folder, 'ctcs_laeq#{}_{}.png'.format(alpha, beta))
    plt.savefig(output_filename)

u = 0.8
dx = 0.1
dt = 0.1
x0 = 0
xmax = 500
t0 = 0
tmax = 1000
tp = 200

# ctcs_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha=0.0, beta=1.0)
# ctcs_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha=0.1, beta=1.0)
ctcs_scheme(x0, xmax, t0, tmax, u, dx, dt, tp, alpha=0.05, beta=0.7)
