import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import os
import math

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

def diffusion_ftcs_animation(ug, vg, f, x0, xmax, t0, tmax, k, dx, dt, tp):
    x = np.arange(x0, xmax + dx, dx)
    n = x.size
    u_now, v_now = u_0(x, ug, vg)
    u_steady, v_steady = u_v_steady(x, f, k, ug, vg)  
    
    # Store trajectories for animation
    trajectories = [(u_now, v_now)]
    times = [t0]
    cmap = cm.coolwarm
    
    plt.figure(figsize=(11,7))
    ax = plt.gca()
    ax.set_xlabel(r'$u$')
    ax.set_ylabel(r'$v$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.4)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, 12)
    plt.tight_layout()
    
    t = t0
    while t <= tmax:
        u_new = np.zeros_like(u_now)
        v_new = np.zeros_like(v_now)

        # BCS
        u_new[0] = 0
        v_new[0] = 0
        u_new[-1] = ug
        v_new[-1] = vg

        for i in range(1, n-1):
            u_new[i] = ((k*dt)/dx**2 )* (u_now[i+1] - 2*u_now[i] + u_now[i-1]) + u_now[i] + dt*f*(v_now[i] - vg)
            v_new[i] = ((k*dt)/dx**2 )* (v_now[i+1] - 2*v_now[i] + v_now[i-1]) + v_now[i] - dt*f*(u_now[i] - ug)

        u_now[:] = u_new[:]
        v_now[:] = v_new[:]
        t += dt
        
        if abs(t % tp) < dt:
            trajectories.append((u_new.copy(), v_new.copy()))
            times.append(t)
    
    # Animation function
    def animate(frame):
        ax.clear()
        ax.set_xlabel(r'$u$')
        ax.set_ylabel(r'$v$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.4)
        ax.set_ylim(-0.5, 4.5)
        ax.set_xlim(0, 12)
        

        # Plot all previous trajectories with color gradient
        for i in range(frame+1):
            norm_t = i / len(trajectories)
            color = cmap(norm_t)
            ax.plot(trajectories[i][0], trajectories[i][1], color=color, alpha=0.7, linewidth=1)
        
        # Current frame in bold
        if frame < len(trajectories):
            ax.plot(trajectories[frame][0], trajectories[frame][1], color='red', linewidth=3, label=f't = {int(times[frame]/3600)} h')
        
        # Steady state
        ax.plot(u_steady, v_steady, 'k--', linewidth=2, label='steady')
        
        title = rf'Parameters: $k$ = {k}, $dx$={dx}, $dt$={round(dt, 2)}'
        ax.set_title(title)
        ax.legend(loc='upper center', frameon=False)
        plt.tight_layout()
    
    anim = FuncAnimation(plt.gcf(), animate, frames=len(trajectories), interval=100, blit=False)
    
    writer = PillowWriter(fps=10)
    fname = os.path.join(output_folder, 'Ekman_spiral.gif')
    anim.save(fname, writer=writer, dpi=100)
    plt.close()
    print(f"Saved {fname}")

# Run animation
f  = 1e-4
ug = 10.0
vg = 0.0
x0=0
xmax=3000
t0=0
tmax=5*86400
k= 5.0
dx=50
tp=5*3600
dt=0.5*(dx**2/(2*k))

diffusion_ftcs_animation(ug, vg, f, x0, xmax, t0, tmax, k, dx, dt, tp)
