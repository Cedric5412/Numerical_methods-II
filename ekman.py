import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import math
# import cmocean

# Creates output folder if it doesn't exist
output_folder = './Plots/'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# Initial condition
def u_0(x, ug, vg):
    u = np.zeros(x.size)
    v = np.zeros(x.size)
    for i in range(x.size):
        u[i] = ug
        v[i] = 0.0*vg
    return u, v

# Steady solution
def u_v_steady(x, f, k, ug, vg):
    d = np.sqrt(f/(2*k))
    u = np.zeros(x.size)
    v = np.zeros(x.size)
    for i in range(x.size):
        u[i] = ug - (ug*np.cos(d*x[i]) + vg*np.sin(d*x[i])) * np.exp(-d*x[i])
        v[i] = vg + (ug*np.sin(d*x[i]) - vg*np.cos(d*x[i])) * np.exp(-d*x[i])
    return u, v


def diffusion_ftcs_scheme(ug, vg, f, x0, xmax, t0, tmax, k, dx, dt, tp):

    x = np.arange(x0, xmax + dx, dx)
    n = x.size
    # L = xmax - x0
    
    u_now, v_now = u_0(x, ug, vg)
    # Steady state:

    u, v = u_v_steady(x, f, k, ug, vg)

    cmap = cm.coolwarm
    plot_count = 0

    fig = plt.figure(figsize=(19,7))
    plt.subplot(121)
    plt.xlabel(r'$u(m.s^{-1})$', fontsize=13)
    plt.ylabel(r'$v(m.s^{-1})$', fontsize=13)
    # plt.axhline(y=0, color='gray', lw=1, alpha=0.7) 
    plt.grid(axis='y', alpha=0.4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplot(122)
    plt.xlabel(r'$(u, v)(m.s^{-1})$', fontsize=13)
    plt.ylabel(r'$z(m)$', fontsize=13)
    plt.plot(u_now, x, color='darkgreen')
    plt.plot(v_now, x, color='darkgreen')
    plt.grid(axis='y', alpha=0.4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    t = t0
    
    while t <= tmax:
        u_new = np.zeros_like(u_now)
        v_new = np.zeros_like(v_now)

        # BCS
        u_new[0] = 0
        v_new[0] = 0

        #  At this height, we assume the wind is already geostrophic
        u_new[-1] = ug
        v_new[-1] = vg


        for i in range(1, n-1):
            u_new[i] = ((k*dt)/dx**2 )* (u_now[i+1] - 2*u_now[i] + u_now[i-1]) + u_now[i] + dt*f*(v_now[i] - vg)
            v_new[i] = ((k*dt)/dx**2 )* (v_now[i+1] - 2*v_now[i] + v_now[i-1]) + v_now[i] - dt*f*(u_now[i] - ug)

        u_now[:] = u_new[:]
        v_now[:] = v_new[:]
        t += dt
        if abs(t % tp) < dt:
            norm_t = t / tmax 
            color = cmap(norm_t)
            plt.subplot(121)
            plt.plot(u_new, v_new, label=f'$t$ = {int(t/3600)} h', color=color)

            plt.subplot(122)
            plt.plot(u_new, x, label=rf'$u({int(t/3600)})$', color=color)
            plt.plot(v_new, x, label=rf'$v({int(t/3600)})$', color=color)
            plot_count += 1

    u0, v0 = u_0(x, ug, vg)
    plt.subplot(121)
    plt.plot(u, v, label='steady', linewidth = 2, color='black')
    plt.plot(u0, v0, marker = 'o', label=f't = {int(t0%3600)}', color = 'darkgreen')
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)  

    plt.subplot(122)
    plt.plot(u, x, linewidth = 2, color='black')
    plt.plot(v, x, linewidth = 2, color='black')

    title = rf'Parameters: $K_m$ = {k}, $\Delta z$={dx}, $dt$={round(dt, 2)}'
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_folder, 'Ekman_spiral.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

f  = 1e-4
ug = 10.0
vg = 0.0
x0=0
xmax=3000
t0=0
tmax=4*86400
k= 5.0
dx=50
tp=5*3600
dt=0.5*(dx**2/(2*k))

diffusion_ftcs_scheme(ug, vg, f, x0, xmax, t0, tmax, k, dx, dt, tp)

