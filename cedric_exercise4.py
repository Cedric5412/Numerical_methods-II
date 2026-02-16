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
        if x[i] < 400 or x[i] > 600:
            y[i] = 0.0
        elif 400 <= x[i] < 500:
            y[i] = 0.1 * (x[i] - 400)
        elif 500 <= x[i] <= 600:
            y[i] = 20.0 - 0.1 * (x[i] - 400)
    return y

def semi_lagrangian(x0=0, xmax=1000, t0=0, tmax=2000, u=0.75, dx=0.5, dt=1.0, tp=250, interp='linear'):
    """
    the interpolation method should be specified as an argument of this function
    possible method: linear, cubic
    """
    x = np.arange(x0, xmax + dx, dx)
    n = x.size
    L = xmax - x0
    
    phi_now = phi_0(x)
    
    plt.figure(figsize=(14,8))
    plt.plot(x, phi_now, label=f't = {int(t0)}')
    plt.xlim(0, 1000)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x, t)$')
    plt.grid(True)
    
    t = t0
    while t <= tmax:
        phi_new = np.zeros_like(phi_now)
        for i in range(n):
            xdep = x0 + (x[i] - u * dt) % L
            
            if xdep < x0:
                xdep = xmax + xdep

            m_float = xdep / dx
            m = int(math.ceil(m_float))
            alpha = m - m_float

            if interp == 'linear':
                m0 = m % n
                m1 = (m - 1) % n
                phi_new[i] = (1 - alpha) * phi_now[m0] + alpha * phi_now[m1]
            
            elif interp == 'cubic':
                m0 = m % n
                m1 = (m - 1) % n
                m2 = (m - 2) % n
                mp1 = (m + 1) % n
                phi_new[i] = (-alpha/6)*(1 - alpha**2) * phi_now[m2] + \
                             (0.5*alpha)*(1 + alpha)*(2 - alpha) * phi_now[m1] + \
                             (0.5*(1 - alpha**2))*(2 - alpha) * phi_now[m0] - \
                             (alpha/6)*(1 - alpha)*(2 - alpha) * phi_now[mp1]
            else:
                print('Interpolation should be either linear or cubic!')
        
        phi_now[:] = phi_new[:]
        t += dt
        if abs(t % tp) < dt:
            plt.plot(x, phi_now, label=f'$t$ = {int(t)}')
            plt.legend()
    
    title = rf'Semilag {interp}: $dx$={dx}, $u$={u}, $\Delta t$={round(dt, 2)}'
    plt.title(r'$\dfrac{\partial \phi}{\partial t} + u \dfrac{\partial \phi}{\partial x} = 0$: ' + title)
    fname = os.path.join(output_folder, f'Semilag_{interp}.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

# Run both
semi_lagrangian(interp='linear')
semi_lagrangian(interp='cubic')
