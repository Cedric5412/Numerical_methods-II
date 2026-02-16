import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import cmocean
# Creates output folder if it doesn't exist
output_folder = './Plots/'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# initial condition
# def phi_0(x, y):
#     """2D Gaussian pulse initial condition. x,y are 1D arrays."""
#     Lx, Ly = 2*np.pi, 2*np.pi
#     x0, y0, sigma = np.pi/2, np.pi, 0.25
#     X, Y = np.meshgrid(x, y, indexing='ij')
#     return np.exp( -((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2) )

def phi_0(x, y):
    """
    2D sinusoidal initial condition for linear advection.

    Parameters
    ----------
    x, y : 1D numpy arrays
        Grid coordinates (assumed periodic and uniform)

    Returns
    -------
    X, Y : 2D arrays (meshgrid)
    phi0 : 2D array
        Initial condition
    """
    Lx = x[-1] - x[0] + (x[1] - x[0])
    Ly = y[-1] - y[0] + (y[1] - y[0])

    X, Y = np.meshgrid(x, y, indexing="ij")

    phi0 = np.sin(2.0 * np.pi * X / Lx) * np.sin(2.0 * np.pi * Y / Ly)

    return phi0

def ftfs(phi_now, c):
    return -c * ( np.roll(phi_now, -1, axis=0) - 2*phi_now + np.roll(phi_now, -1, axis=1) ) + phi_now

def ftbs(phi_now, c):
    return (1-2*c)*phi_now + c*(np.roll(phi_now, 1, axis=0) + np.roll(phi_now, 1, axis=1))

def ctcs(phi_now, phi_old, c):
    return phi_old - c * ( np.roll(phi_now, -1, axis=0) - np.roll(phi_now, 1, axis=0)) -c*( np.roll(phi_now, -1, axis=1) - np.roll(phi_now, 1, axis=1) )   

def ctcs_scheme(x0, xmax, y0, ymax, t0, tmax, u, d, dt, alpha, beta):
    
    x = np.arange(x0, xmax+d, d)
    y = np.arange(y0, ymax+d, d)

    # Need this for upwind scheme
    ux = u/np.sqrt(2.0)            # velocity along x, y
    uy = ux
    cx =  ux*dt/d
    cy =  ux*dt/d

    phi0 = phi_0(x, y)
    phi1 = np.zeros_like(phi0)

    if ux < 0:
        phi1 = ftfs(phi0, cx)
    else :
        phi1 = ftbs(phi0, cx)

    phi_fields = [phi0.copy()]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initial display
    ax.imshow(phi0, cmap=cmocean.cm.deep, extent=[0,2*np.pi,0,2*np.pi], origin='lower')
    ax.set_title('t = 0')
    
    t = t0 + dt
    while t<= tmax:
        phi_new = np.zeros_like(phi1)
        phi_new = ctcs(phi1, phi0, cx)

        d = alpha*(phi0 - 2*phi1 + phi_new)
        phi1_smoothed = phi1 + beta*d
        phi_new_smoothed = phi_new + (1-beta)*d

        phi0 = phi1_smoothed
        phi1 = phi_new_smoothed
        t += dt
        
        phi_fields.append(phi_new.copy())  # Store for GIF

        ax.clear()
        ax.imshow(phi_new, cmap=cmocean.cm.deep, extent=[0,2*np.pi,0,2*np.pi], origin='lower')
        ax.set_title(f't = {t:.1f}')

    # Create animated GIF from stored fields
    def animate(frame):
        ax.clear()
        ax.imshow(phi_fields[frame], cmap=cmocean.cm.deep, extent=[0,2*np.pi,0,2*np.pi], origin='lower')
        ax.set_title(f't = {frame*dt:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    anim = FuncAnimation(fig, animate, frames=len(phi_fields), interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    filename = f'./Plots/ctcs_2D_alpha{alpha}_beta{beta}.gif'
    anim.save(filename, writer=writer)
    print(f"Saved: {filename}")
    plt.close(fig)

u = 1
x0 = 0
xmax = 2*np.pi
y0 = x0
ymax = xmax
d = (xmax-x0)/500 
dt = d/(u*np.sqrt(2))  # stability condition
t0 = 0
tmax = 10

ctcs_scheme(x0, xmax, y0, ymax, t0, tmax, u, d, dt, alpha=0.0, beta=1)
