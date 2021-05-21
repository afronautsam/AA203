import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.animation as animation




def Riccati(R, A, B, Q, PkOld):
    Kk = -inv(R + B.T@PkOld@B)@B.T@PkOld@A
    Pk = Q + A.T@PkOld@(A + B@Kk)

    while LA.norm(Pk - PkOld) > 1e-4:
        PkOld = Pk
        Kk = -inv(R + B.T@PkOld@B)@B.T@PkOld@A
        Pk = Q + A.T@PkOld@(A + B@Kk)

    return Pk, Kk

def cartpole(s, t, u):
    mP = 2
    mC = 10
    l = 1
    g = 9.81
    dT = 0.1

    x = s[0]
    theta = s[1]
    xDot = s[2]
    thetaDot = s[3]

    ds = np.zeros(4)
    ds[0] = xDot
    ds[1] = thetaDot
    ds[2] = (mP*np.sin(theta)*(l*(thetaDot**2) + g*np.cos(theta)) + u)/(mC + mP*(np.sin(theta)**2))
    ds[3] = -((mC + mP)*g*np.sin(theta) + mP*l*(thetaDot**2)*np.sin(theta)*np.cos(theta) + u*np.cos(theta))/((mC + mP*(np.sin(theta)**2))*l)

    return ds

def simulate(kInf, sInit, noisemod):
    t = np.linspace(0, 30, 300)
    s = np.zeros((300, 4))
    u = np.zeros(300)
    sOpt = np.array([0, np.pi, 0, 0])
    s[0] = sInit
    #u[0] = kInf@s[0]
    u[0] = kInf@np.zeros(4)
    if noisemod == True:
        mu = [0, 0, 0, 0]
        sigma = np.diag([0, 0, 1e-4, 1e-4])
        for k in range(len(t)-1):
            w = np.random.multivariate_normal(mu, sigma, 1)
            w = w.flatten()
            u[k] = u[k] = kInf@(s[k]+w - sOpt)
            s[k+1] = odeint(cartpole, s[k]+w, t[k:k+2], args=(u[k],))[1]
            #u[k+1] = kInf@(s[k+1] - s[k])
    
    else:
        for k in range(len(t)-1):
            u[k] = kInf@(s[k] - sOpt)
            s[k+1] = odeint(cartpole, s[k], t[k:k+2], args=(u[k],))[1]
            #u[k+1] = kInf@(s[k+1] - s[k])
    

    

    return s,t


def animate_cartpole(t, x, θ):
    """Animate the cart-pole system from given position data.

    The arguments `t`, `x`, and `θ` are assumed to be 1-D Numpy arrays
    describing the degrees of freedom (i.e., `x` and `θ`) of the cart-pole over
    time (i.e., `t`).

    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate_cartpole
        fig, ani = animate_cartpole(t, x, θ)
        ani.save('cartpole_balance.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    cart_width = 2.
    cart_height = 1.
    wheel_radius = 0.3
    wheel_sep = 1.
    pole_length = 5.
    mass_radius = 0.25

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_lim = 1.1*np.max(np.abs(x))
    y_lim = 1.1*(wheel_radius + cart_height + pole_length)
    ax.set_xlim([-1.1*x_lim, 1.1*x_lim])
    ax.set_ylim([0., y_lim])
    ax.set_yticks([])
    ax.set_aspect(1.)

    # Artists
    cart = mpatches.FancyBboxPatch((0., 0.), cart_width, cart_height,
                                   facecolor='tab:blue', edgecolor='k',
                                   boxstyle='Round,pad=0.,rounding_size=0.05')
    wheel_left = mpatches.Circle((0., 0.), wheel_radius, color='k')
    wheel_right = mpatches.Circle((0., 0.), wheel_radius, color='k')
    mass = mpatches.Circle((0., 0.), mass_radius, color='k')
    pole = ax.plot([], [], '-', linewidth=3, color='k')[0]
    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(-x_lim, 0.8*y_lim, '')

    ax.add_patch(cart)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)
    ax.add_patch(mass)

    def animate(k, t, x, θ):
        # Geometry
        cart_corner = np.array([x[k] - cart_width/2, wheel_radius])
        wheel_left_center = np.array([x[k] - wheel_sep/2, wheel_radius])
        wheel_right_center = np.array([x[k] + wheel_sep/2, wheel_radius])
        pole_start = np.array([x[k], wheel_radius + cart_height])
        pole_end = pole_start + pole_length*np.array([np.sin(θ[k]),
                                                      -np.cos(θ[k])])

        # Cart
        cart.set_x(cart_corner[0])
        cart.set_y(cart_corner[1])

        # Wheels
        wheel_left.set_center(wheel_left_center)
        wheel_right.set_center(wheel_right_center)

        # Pendulum
        pole.set_data([pole_start[0], pole_end[0]],
                      [pole_start[1], pole_end[1]])
        mass.set_center(pole_end)
        mass_x = x[:k+1] + pole_length*np.sin(θ[:k+1])
        mass_y = wheel_radius + cart_height - pole_length*np.cos(θ[:k+1])
        trace.set_data(mass_x, mass_y)

        # Time-stamp
        timestamp.set_text('t = {:.1f} s'.format(t[k]))

        artists = (cart, wheel_left, wheel_right, pole, mass, trace, timestamp)
        return artists

    dt = t[1] - t[0]
    ani = animation.FuncAnimation(fig, animate, t.size, fargs=(t, x, θ),
                                  interval=dt*1000, blit=True)
    return fig, ani



def main():
    mP = 2 #in kg
    mC = 10 # in Kg
    l = 1 #in m
    g = 9.81 #in m/s^2
    dT = 0.1 #in seconds 
    Q = np.identity(4)
    R = np.identity(1)
    initP = np.zeros((4,4))
    A = np.array([[1, 0, dT, 0], [0, 1, 0 ,dT], [0, (dT*mP*g)/mC, 1, 0], [0, (dT*(mC + mP)*g)/(mC*l), 0, 1]])
    B = np.array([[0], [0], [dT/mC], [dT/(mC*l)]])

    Pk, Kk = Riccati(R, A, B, Q, initP)

    sInit = np.array([0.2, 3*(np.pi)/4, 0, 0])
    
    stateInt, timesteps = simulate(Kk, sInit, noisemod=False)
    noisyStateInt, timesteps = simulate(Kk, sInit, noisemod=True)

    fig, ani = animate_cartpole(timesteps, stateInt[:,0], stateInt[:,1])
    ani.save('cartpole_balance.gif', writer='ffmpeg')
    plt.show()

    # fig1,ax1 = plt.subplots()
    # ax1.plot(timesteps, stateInt[:,0],label='x')
    # ax1.plot(timesteps, stateInt[:,1], label='theta')
    # ax1.plot(timesteps, stateInt[:,2], label='xDot')
    # ax1.plot(timesteps, stateInt[:,3], label='thetaDot')
    # ax1.set_xlabel('Timesteps (in seconds)')
    # ax1.set_ylabel('State Variable Magnitude')
    # ax1.legend()
    # ax1.set_title('LQR Controller with Noise-Free Dynamics')
    # #ax1.plot(timesteps, stateInt[:,0])


    # fig2,ax2 = plt.subplots()
    # ax2.plot(timesteps, noisyStateInt[:,0],label='x')
    # ax2.plot(timesteps, noisyStateInt[:,1], label='theta')
    # ax2.plot(timesteps, noisyStateInt[:,2], label='xDot')
    # ax2.plot(timesteps, noisyStateInt[:,3], label='thetaDot')
    # ax2.set_xlabel('Timesteps (in seconds)')
    # ax2.set_ylabel('State Variable Magnitude')
    # ax2.legend()
    # ax2.set_title('LQR Controller with Noisy Dynamics')


    # plt.show()
    




    ray = 2 + 1



if __name__ == "__main__":
    main()    


