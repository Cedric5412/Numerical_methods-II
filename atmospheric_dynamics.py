import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cmocean
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
# ----------------------
# Create output folder
# ----------------------
output_folder = './Plots/'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# ----------------------
# PARAMETERS
# ----------------------
Lx = 2e7
Ly = 3e6
d = 1e5
H = 1.2e4
h0 = 1e3
beta = 1.6e-11
f0 = 1e-4
re = 1/(24*3600)
dt = 3600                # 1 hour
tmax = 7*24*3600         # 7 days

nx = int(Lx/d)
ny = int(Ly/d) + 1

x = np.linspace(0, Lx-d, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# ----------------------
# MODEL FUNCTION
# ----------------------
def run_model_anim(N):
    hT = h0*np.sin(2*N*np.pi*X/Lx)*np.sin(np.pi*Y/Ly)
    h = H - hT

    psi = -10*(Y - Ly)        # initial streamfunction
    vort = np.zeros_like(psi)
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    F = np.zeros_like(psi)

    psi_fields = [psi.copy()]  # store for animation
    times = [0]

    t = 0
    while t < tmax:

        # ---- Boundary conditions ----
        psi[:,0]  = 10*Ly   # constant at bottom wall
        psi[:,-1] = 0       # constant at top wall

        # ---- Compute vorticity and velocity ----
        for i in range(nx):
            ip = (i+1) % nx
            im = (i-1) % nx
            for j in range(1, ny-1):
                vort[i,j] = (psi[ip,j] + psi[im,j] +
                             psi[i,j+1] + psi[i,j-1] -
                             4*psi[i,j]) / d**2
                u[i,j] = -(psi[i,j+1] - psi[i,j-1])/(2*d)
                v[i,j] =  (psi[ip,j] - psi[im,j])/(2*d)

        # ---- Compute forcing F ----
        for i in range(nx):
            ip = (i+1) % nx
            im = (i-1) % nx
            for j in range(1, ny-1):
                adv = -( (u[ip,j]*vort[ip,j] - u[im,j]*vort[im,j])
                       + (v[i,j+1]*vort[i,j+1] - v[i,j-1]*vort[i,j-1]) )/(2*d)
                beta_term = -beta * v[i,j]
                stretch = (f0/h[i,j]) * (
                         (u[ip,j]*h[ip,j] - u[im,j]*h[im,j])
                       + (v[i,j+1]*h[i,j+1] - v[i,j-1]*h[i,j-1])
                         )/(2*d)
              
                F[i,j] = adv + beta_term + stretch 

        # ---- Time step (implicit Ekman) ----
        vort = (vort + dt*F) / (1 + dt*re)

        # ---- Invert Laplacian (SOR) ----
        psi_new = psi.copy()
        omega = 1.7
        tol = 1e-6

        for it in range(3000):
            err = 0
            for i in range(nx):
                ip = (i+1) % nx
                im = (i-1) % nx
                for j in range(1, ny-1):
                    old = psi_new[i,j]
                    gs = 0.25*(psi_new[ip,j] + psi_new[im,j] +
                               psi_new[i,j+1] + psi_new[i,j-1]
                               - d**2*vort[i,j])
                    psi_new[i,j] = (1-omega)*old + omega*gs
                    err = max(err, abs(psi_new[i,j]-old))
            psi_new[:,0]  = 10*Ly
            psi_new[:,-1] = 0
            if err < tol:
                break

        psi = psi_new.copy()
        t += dt

        # Store fields every 2 hours
        if t % (2*3600) == 0:
            psi_fields.append(psi.copy())
            times.append(t/3600)

    return psi_fields, hT, times

# ----------------------
# RUN MODEL N=2
# ----------------------
psi_fields, hT, times = run_model_anim(N=8)

# ----------------------
# CREATE ANIMATION
# ----------------------
fig, ax = plt.subplots(figsize=(12,5))

def animate(frame):
    ax.clear()
    cs = ax.contour(X/1e3, Y/1e3, hT, levels=10, colors='k', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%1.0f')
    cf = ax.contourf(X/1e3, Y/1e3, psi_fields[frame], levels=20, cmap=cmocean.cm.balance)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  
    cbar = plt.colorbar(cf, cax=cax)
    cbar.set_label(r'$ \psi \;[m^2/s]$', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(f'Time = {times[frame]:.1f} hours (N=8)')
    # ax.set_aspect('equal')
    return cf

anim = FuncAnimation(fig, animate, frames=len(psi_fields), interval=200, blit=False)

# Save GIF
filename = os.path.join(output_folder, f'psi_N8_7days.gif')
writer = PillowWriter(fps=10)
anim.save(filename, writer=writer)
print(f"Saved: {filename}")
plt.close(fig)
