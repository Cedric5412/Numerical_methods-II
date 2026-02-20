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
# PARAMETERS (as given in problem, but shortened run)
# ----------------------
Lx = 2e7           # m
Ly = 3e6           # m
d  = 1e5           # m
H  = 1.2e4         # m
h0 = 1e3           # m
beta = 1.6e-11     # m⁻¹ s⁻¹
f0   = 1e-4        # s⁻¹
re   = 1/(24*3600) # s⁻¹  (= 1/day Ekman damping)
dt   = 3600        # s   (= 1 hour)
tmax = 7*24*3600   # s   ← changed to 2 days

nx = int(Lx / d)
ny = int(Ly / d) + 1

x = np.linspace(0, Lx - d, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# ----------------------
# MODEL FUNCTION
# ----------------------
def run_model_anim(N):
    # Topography
    hT = h0 * np.sin(2 * N * np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    h  = H - hT

    # Initial streamfunction = zonal flow U = 10 m/s eastward
    psi = -10 * (Y - Ly)          # → u = -∂ψ/∂y = 10 m/s, v = ∂ψ/∂x = 0

    vort = np.zeros_like(psi)
    u    = np.zeros_like(psi)
    v    = np.zeros_like(psi)

    psi_fields = [psi.copy()]
    times      = [0]

    t = 0.0
    step = 0

    while t < tmax:
        # 1. Enforce ψ boundary conditions (reflecting walls)
        psi[:,  0] = 10 * Ly     # bottom wall
        psi[:, -1] = 0.0         # top wall

        # 2. Compute interior vorticity ξ = ∇²ψ
        for i in range(nx):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            for j in range(1, ny - 1):
                vort[i, j] = (psi[ip, j]     + psi[im, j] +
                              psi[i, j + 1]  + psi[i, j - 1] -
                              4 * psi[i, j]) / d**2

        # 3. Set boundary vorticity (free-slip condition)
        for i in range(nx):
            # South wall (j=0) — one-sided difference
            vort[i, 0] = (psi[i, 2] - 2 * psi[i, 1] + psi[i, 0]) / d**2
            
            # North wall (j=ny-1)
            vort[i, ny-1] = (psi[i, ny-3] - 2 * psi[i, ny-2] + psi[i, ny-1]) / d**2

        # 4. Compute velocities (interior)
        for i in range(nx):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            for j in range(1, ny - 1):
                u[i, j] = -(psi[i, j + 1] - psi[i, j - 1]) / (2 * d)
                v[i, j] =  (psi[ip, j]     - psi[im, j])     / (2 * d)

        # Set wall velocities (v=0, u one-sided)
        for i in range(nx):
            v[i, 0]     = 0.0
            v[i, ny-1]  = 0.0
            u[i, 0]     = -(psi[i, 1]     - psi[i, 0])     / d   # one-sided
            u[i, ny-1]  = -(psi[i, ny-1]  - psi[i, ny-2])  / d   # one-sided

        # 5. Compute forcing F
        F = np.zeros_like(psi)
        for i in range(nx):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            for j in range(1, ny - 1):
                adv = -0.5/d * (
                    (u[ip, j] * vort[ip, j] - u[im, j] * vort[im, j]) +
                    (v[i, j + 1] * vort[i, j + 1] - v[i, j - 1] * vort[i, j - 1])
                )
                beta_term  = -beta * v[i, j]
                stretch    = (f0 / h[i, j]) * 0.5/d * (
                    (u[ip, j] * h[ip, j] - u[im, j] * h[im, j]) +
                    (v[i, j + 1] * h[i, j + 1] - v[i, j - 1] * h[i, j - 1])
                )
                F[i, j] = adv + beta_term + stretch

        # 6. Time step vorticity (implicit Ekman)
        vort = (vort + dt * F) / (1 + dt * re)

        # 7. Invert Laplacian: solve ∇²ψ = vort using SOR
        psi_new = psi.copy()
        omega = 1.78
        tol   = 1e-6
        maxit = 5000

        for it in range(maxit):
            err = 0.0
            for i in range(nx):
                ip = (i + 1) % nx
                im = (i - 1) % nx
                for j in range(1, ny - 1):
                    old = psi_new[i, j]
                    gs  = 0.25 * (psi_new[ip, j] + psi_new[im, j] +
                                  psi_new[i, j + 1] + psi_new[i, j - 1]
                                  - d**2 * vort[i, j])
                    psi_new[i, j] = (1 - omega) * old + omega * gs
                    err = max(err, abs(psi_new[i, j] - old))

            # Enforce ψ BCs every iteration
            psi_new[:,  0] = 10 * Ly
            psi_new[:, -1] = 0.0

            if err < tol:
                break

        psi = psi_new.copy()

        t += dt
        step += 1

        # Store every 1 hour (more frames for short 2-day run)
        if step % 1 == 0:
            psi_fields.append(psi.copy())
            times.append(t / 3600)

    return psi_fields, hT, times


# ----------------------
# RUN for N=2 and N=8 (2 days)
# ----------------------
print("Running N=2 (2 days)...")
psi_fields_2, hT_2, times_2 = run_model_anim(N=2)

# print("Running N=8 (2 days)...")
# psi_fields_8, hT_8, times_8 = run_model_anim(N=8)


# ----------------------
# ANIMATION FUNCTION
# ----------------------
def make_animation(psi_fields, hT, times, N, filename_suffix="2days"):
    fig, ax = plt.subplots(figsize=(14, 6))

    def animate(frame):
        ax.clear()
        # Topography contours
        cs = ax.contour(X/1e3, Y/1e3, hT, levels=10, colors='k', linewidths=0.9)
        ax.clabel(cs, inline=True, fontsize=9, fmt='%1.0f')

        # Streamfunction (total ψ — includes mean flow)
        levels = np.linspace(-1.5e6, 1.5e6, 31)
        cf = ax.contourf(X/1e3, Y/1e3, psi_fields[frame],
                         levels=levels, cmap=cmocean.cm.balance, extend='both')

        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.06)
        cbar = plt.colorbar(cf, cax=cax)
        cbar.set_label(r'$\psi$  [m²/s]', fontsize=12)

        ax.set_xlabel('x  (km)')
        ax.set_ylabel('y  (km)')
        ax.set_title(f'N = {N}   —   Time = {times[frame]:.1f} hours (2-day run)')
        return cf,

    anim = FuncAnimation(fig, animate, frames=len(psi_fields),
                         interval=150, blit=False)

    filename = os.path.join(output_folder, f'psi_N{N}_2days_{filename_suffix}.gif')
    anim.save(filename, writer=PillowWriter(fps=12))
    print(f"Saved: {filename}")
    plt.close(fig)


make_animation(psi_fields_2, hT_2, times_2, N=2)
# make_animation(psi_fields_8, hT_8, times_8, N=8)