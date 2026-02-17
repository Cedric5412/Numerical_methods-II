import numpy as np
import matplotlib.pyplot as plt
import os

# Creates output folder if it doesn't exist
output_folder = './Plots/'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# initial condition
def phi_0(x):
    return np.array([
        np.sin(np.pi * ((xi - 400)/200))**2 if 400 <= xi < 600 else 0.0
        for xi in x
    ])

def u_0(x):
    return np.array([0.0 for xi in x])

def ftcs_wave(u_now, phi_now, dtdx, c):
    u_new = u_now - 0.5*dtdx*(np.roll(phi_now, -1) - np.roll(phi_now, 1)) 
    phi_new = phi_now - 0.5*c* (np.roll(u_new, -1) - np.roll(u_new, 1))
    return u_new, phi_new

def ctcs_wave(u_old, phi_old, u_now, phi_now, dtdx, c):
    u_new = u_old - dtdx*(np.roll(phi_now, -1) - np.roll(phi_now, 1)) 
    phi_new = phi_old - c* (np.roll(u_now, -1) - np.roll(u_now, 1))
    return u_new, phi_new


#=================== FTCS ================================================
def ftcs_wave_solver(x0, xmax, t0, tmax, P, dx, dt, tp):
    x = np.arange(x0, xmax+dx, dx)

    dtdx = dt/dx
    c = np.sqrt(P) * dtdx
    
    phi_now = phi_0(x)
    u_now   = u_0(x)

    fig, ax = plt.subplots(1,2,figsize=(14,6),sharey=False)

    ax[0].plot(x, phi_now, label=r'$\phi({})$'.format(int(t0)))
    ax[1].plot(x, u_now, label=r'$u({})$'.format(int(t0)))

    ax[0].grid(True)
    ax[1].grid(True)

    t = t0 
    while t<= tmax:
        phi_new = np.zeros_like(phi_now)
        u_new = np.zeros_like(u_now)

        u_new, phi_new = ftcs_wave(u_now, phi_now, dtdx, c)

        # update u, phi 
        u_now, phi_now = u_new, phi_new
 
        t += dt
    
        if t%tp < dt: 
            ax[0].plot(x, phi_new, label=r'$\phi({})$'.format(int(t)))
            ax[1].plot(x, u_new, label=r'$u({})$'.format(int(t)))
            ax[0].legend(fontsize=14, loc='upper right')
            ax[1].legend(fontsize=14, loc='upper right')
    plt.suptitle(r'Gravity waves: $u(x, 0) = 0$, $\Delta t = {}$'.format(dt))
    output_filename = os.path.join(output_folder, 'wave_ftcs.png')
    plt.tight_layout()
    plt.savefig(output_filename)


# =============== CTCS SCHEME ==========================================
def ctcs_wave_solver(x0, xmax, t0, tmax, P, dx, dt, tp):
    x = np.arange(x0, xmax+dx, dx)

    dtdx = dt/dx
    c = np.sqrt(P) * dtdx
    
    phi_old = phi_0(x)
    u_old   = u_0(x)

    phi_now = np.zeros_like(phi_old)
    u_now   = np.zeros_like(u_old)

    # Use FTCS for the first step then switch to CTCS in the remaining steps.
    u_now, phi_now = ftcs_wave(u_old, phi_old, dtdx, c)

    fig, ax = plt.subplots(1,2,figsize=(14,6),sharey=False)

    ax[0].plot(x, phi_now, label=r'$\phi({})$'.format(int(t0)))
    ax[1].plot(x, u_now, label=r'$u({})$'.format(int(t0)))

    ax[0].grid(True)
    ax[1].grid(True)

    t = t0 + dt
    while t<= tmax:
        phi_new = np.zeros_like(phi_now)
        u_new = np.zeros_like(u_now)

        u_new, phi_new = ctcs_wave(u_old, phi_old, u_now, phi_now, dtdx, c)

        # update phi, u 
        u_old, phi_old = u_now, phi_now
        u_now, phi_now = u_new, phi_new
 
        t += dt
    
        if t%tp < dt: 
            ax[0].plot(x, phi_new, label=r'$\phi({})$'.format(int(t)))
            ax[1].plot(x, u_new, label=r'$u({})$'.format(int(t)))
            ax[0].legend(fontsize=14, loc='upper right')
            ax[1].legend(fontsize=14, loc='upper right')
    plt.suptitle(r'Gravity waves: $u(x, 0) = 0$, $\Delta t = {}$'.format(dt))
    output_filename = os.path.join(output_folder, 'wave_ctcs.png')
    plt.tight_layout()
    plt.savefig(output_filename)


P = 1
dx = 0.5
x0 = 0
xmax = 1000
t0 = 0
tmax = 2000
tp = 200

'''
Choose dt s.t.:
        1.  sqrt(gH)*(dt/dx) < 2 for the FTCS      
        2.  sqrt(gH)*(dt/dx) <= 1 for the CTCS

=======>> These schemes requires different dt ;  
          And FTCS allows us to use larger timestep.
'''
 
dt = 0.5

ftcs_wave_solver(x0, xmax, t0, tmax, P, dx, dt, tp)
ctcs_wave_solver(x0, xmax, t0, tmax, P, dx, dt, tp)