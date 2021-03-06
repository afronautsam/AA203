import numpy as np
from scipy.integrate import odeint
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import cvxpy as cvx
import imageio
import os


@jax.partial(jax.jit, static_argnums=(0,))
def linearize(f, x, u):
    A = jax.jacfwd(f, 0)(x, u)
    B = jax.jacfwd(f, 1)(x, u)
    c = f(x, u)
    return A, B, c


def plotReachBot(s,u,t,HL):
    CMvec = []
    xCMvec = []
    yCMvec = []
    phivec = []
    psivec = []
    F1vec = []
    M1vec = []
    M4vec = []
    for (sk,uk) in zip(s,u):
        xCM, yCM, phi, psi = state_to_CM(sk,HL)
        CMvec.append(np.array([xCM,yCM]))
        xCMvec.append(xCM)
        yCMvec.append(yCM)
        phivec.append(phi)
        psivec.append(psi)
        F1vec.append(uk[0])
        M1vec.append(uk[1])
        M4vec.append(uk[2])
    fig1 = plt.figure(1)
    plt.plot(xCMvec,yCMvec)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.title('ReachBot Optimal Trajectory')
    fig2 = plt.figure(2)
    plt.plot(t[0:-1], F1vec, label='Boom 1 Force')
    plt.plot(t[0:-1], M1vec, label='Boom 1 Torque')
    plt.plot(t[0:-1], M4vec, label='Boom 4 Torque')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Force [N] and Torque [N.m]')
    plt.title('ReachBot Optimal Control')
    plt.grid()
    plt.legend()
    plt.show()
    return xCMvec, yCMvec, phivec, psivec


def gifReachBot(xCMvec, yCMvec, phivec, psivec, W, HL, HR, delH1, s):
    b = 0.25  # hub side length
    fig3 = plt.figure(3)
    filenames = []
    minH = np.min(np.array([0, delH1]))
    maxH = np.max(np.array([HL, delH1 + HR]))
    x_leftwall = np.array([0, 0])
    y_leftwall = np.array([minH, maxH])
    x_rightwall = np.array([W, W])
    y_rightwall = np.array([minH, maxH])
    llboom = np.array([0, 0])
    ulboom = np.array([0, HL])
    lrboom = np.array([W, delH1])
    urboom = np.array([W, delH1 + HR])
    for i, (xCM, yCM, phi, psi, sk) in enumerate(zip(xCMvec, yCMvec, phivec, psivec, s)):
        ax3 = plt.gca()
        ax3.set_axisbelow(True)
        l4 = getl4(sk[0],sk[2],sk[4])
        th4 = sk[4]
        llc = np.array([xCM-(b*np.sqrt(2)/2)*np.cos(psi), yCM-(b*np.sqrt(2)/2)*np.sin(psi)])
        ulc = np.array([xCM-(b*np.sqrt(2)/2)*np.sin(psi), yCM+(b*np.sqrt(2)/2)*np.cos(psi)])
        lrc = np.array([xCM+(b*np.sqrt(2)/2)*np.sin(psi), yCM-(b*np.sqrt(2)/2)*np.cos(psi)])
        urc = np.array([xCM+(b*np.sqrt(2)/2)*np.cos(psi), yCM+(b*np.sqrt(2)/2)*np.sin(psi)])
        lrc = np.array([W-l4*np.cos(th4),delH1+l4*np.sin(th4)])
        RBot = Rectangle(xy=(xCM-(b*np.sqrt(2)/2)*np.cos(psi), yCM-(b*np.sqrt(2)/2)*np.sin(psi)), height=b, width=b, angle=(-phi*180/np.pi), edgecolor='0.4', facecolor='0.4')
        plt.plot(x_leftwall, y_leftwall, 'k', linewidth=2)
        plt.plot(x_rightwall, y_rightwall, 'k', linewidth=2)
        ax3.add_patch(RBot)
        plt.plot([llboom[0], llc[0]], [llboom[1], llc[1]], '0.4')
        plt.plot([ulboom[0], ulc[0]], [ulboom[1], ulc[1]], '0.4')
        plt.plot([lrboom[0], lrc[0]], [lrboom[1], lrc[1]], '0.4')
        plt.plot([urboom[0], urc[0]], [urboom[1], urc[1]], '0.4')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid()
        plt.title('ReachBot Trajectory')
        plt.savefig(str(i) + '.png')
        filenames.append(str(i) + '.png')
        plt.cla()
    with imageio.get_writer('ReachBot2D.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(filenames):
        os.remove(filename)


def CM_to_state(CMvec, HL, HR, delH1, W):
    b = 0.25  # hub side length
    xCM, yCM = CMvec
    l1 = np.sqrt(np.square(0 - (xCM - b / 2)) + np.square(HL - (yCM + b / 2)))
    th1 = np.arccos((xCM - b / 2) / l1)
    l4 = np.sqrt(np.square(W - (xCM + b / 2)) + np.square(delH1 - (yCM - b / 2)))
    th4 = np.arccos((W - (xCM + b / 2)) / l4)
    statevec = np.array([l1, 0., th1, 0., th4, 0.])
    return statevec


def state_to_CM(s, HL):
    b = 0.25  # hub side length
    l1, dl1, th1, dth1, th4, dth4 = s
    l4 = getl4(l1,th1,th4)  # find l4 that ensures ReachBot shape is retained
    psi = np.arcsin((W-l1*np.cos(th1)-l4*np.cos(th4))/(b*np.sqrt(2)))  # corner heading angle
    phi = np.pi/4. - psi  # CM heading angle
    xCM = l1*np.cos(th1) + (b*np.sqrt(2)/2)*np.sin(psi)
    yCM = HL - l1*np.sin(th1) - (b*np.sqrt(2)/2)*np.cos(psi)
    return xCM, yCM, phi, psi


def CM_to_full(CMvec, HL, HR, delH1, W):
    b = 0.25  # hub side length
    xCM, yCM = CMvec
    l1 = np.sqrt(np.square(0 - (xCM - b / 2)) + np.square(HL - (yCM + b / 2)))
    th1 = np.arccos((xCM - b / 2) / l1)
    l2 = np.sqrt(np.square(W - (xCM + b / 2)) + np.square(HR - (yCM - delH1 + b / 2)))
    th2 = np.arccos((W - (xCM + b / 2)) / l2)
    l3 = np.sqrt(np.square(0 - (xCM - b / 2)) + np.square(0 - (yCM - b / 2)))
    th3 = np.arccos((xCM - b / 2) / l3)
    l4 = np.sqrt(np.square(W - (xCM + b / 2)) + np.square(delH1 - (yCM - b / 2)))
    th4 = np.arccos((W - (xCM + b / 2)) / l4)
    full_lvec = np.array([l1, l2, l3, l4])
    full_thvec = np.array([th1, th2, th3, th4])
    return full_lvec, full_thvec


def getl4(l1,th1,th4):
    b = 0.25  # hub side length
    C1 = 1
    C2 = 2*delH1*np.sin(th4) - 2*HL*np.sin(th4) + 2*l1*np.sin(th1)*np.sin(th4) + 2*l1*np.cos(th1)*np.cos(th4) - 2*W*np.cos(th4)
    C3 = np.square(delH1) - 2*delH1*HL + 2*delH1*l1*np.sin(th1) + np.square(HL) - 2*HL*l1*np.sin(th1) + np.square(l1) - 2*l1*W*np.cos(th1) + np.square(W) - 2*np.square(b)
    poly = np.array([C1, C2, C3])
    lengths = np.roots(poly)
    for lposs in lengths:
        if lposs*np.sin(th4) + l1*np.sin(th1) < HL-delH1 + 0.1:
            l4 = lposs
    return l4


def set_bounds(CM1, CM2, HL, HR, delH1, W):
    # establish reasonable extrema
    llb = 0.5
    lub = 4.
    thlb = -np.pi/6
    thub = 75.*np.pi/180.

    # find start/goal extrema
    full_lvec1, full_thvec1 = CM_to_full(CM1, HL, HR, delH1, W)
    full_lvec2, full_thvec2 = CM_to_full(CM2, HL, HR, delH1, W)
    lvec = np.concatenate((full_lvec1,full_lvec2))
    thvec = np.concatenate((full_thvec1,full_thvec2))
    lmin = np.min(lvec)
    lmax = np.max(lvec)
    thmin = np.min(thvec)
    thmax = np.max(thvec)

    # adjust extrema to return
    if llb > lmin:
        llb = lmin
    if lub < lmax:
        lub = lmax
    if thlb > thmin:
        thlb = thmin
    if thub < thmax:
        thub = thmax

    return llb, lub, thlb, thub


def scp(f, Q, R, Q_N, s_star, s0, N, dt, rho, lLB, lUB, thLB, thUB):
    # Outer loop of scp
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension
    eps = 0.01  # termination threshold for scp

    # initialize reference rollout s_bar,u_bar
    u_bar = np.zeros((N, m))

    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k + 1] = f(s_bar[k], u_bar[k])

    # Compute new state and control via scp
    s, u = scp_iteration(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, lLB, lUB, thLB, thUB)

    # run scp until u converges
    round = 0
    while (np.linalg.norm(u - u_bar, np.inf) > eps):
        print("round: %s, u update: %s" % (round, np.linalg.norm(u - u_bar, np.inf)))
        round = round + 1
        s_bar = s
        u_bar = u
        s, u = scp_iteration(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, lLB, lUB, thLB, thUB)
    return s, u


def scp_iteration(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, lLB, lUB, thLB, thUB):
    X = {}
    U = {}
    cost_terms = []
    constraints = []
    A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)
    for k in range(N):
        X[k] = cvx.Variable(6)
        U[k] = cvx.Variable(3)
        cost_terms.append(cvx.quad_form(X[k] - s_star, Q))
        cost_terms.append(cvx.quad_form(U[k], R))
        constraints.append(X[k][0] <= lUB)
        constraints.append(X[k][0] >= lLB)
        constraints.append(X[k][2] <= thUB)
        constraints.append(X[k][2] >= thLB)
        constraints.append(X[k][4] <= thUB)
        constraints.append(X[k][4] >= thLB)
        constraints.append(cvx.norm(U[k][1]) <= 1.)  # 1 N.m maximum torque
        constraints.append(cvx.norm(U[k][0]) <= 1.)  # 1 N maximum force
        constraints.append(cvx.norm(U[k][2]) <= 1.)  # 1 N maximum torque
        constraints.append(cvx.norm(U[k] - u_bar[k], "inf") <= rho)
        constraints.append(cvx.norm(X[k] - s_bar[k], "inf") <= rho)
        if (k == 0):
            constraints.append(X[k] == s0)
        if (k < N and k > 0):
            constraints.append(
                A[k - 1] @ (X[k - 1] - s_bar[k - 1]) + B[k - 1] @ (U[k - 1] - u_bar[k - 1]) + c[k - 1] == X[k])
    X[N] = cvx.Variable(6)
    cost_terms.append(cvx.quad_form(X[N] - s_star, Q_N))
    constraints.append(X[N][0] <= lUB)
    constraints.append(X[N][0] >= lLB)
    constraints.append(X[N][2] <= thUB)
    constraints.append(X[N][2] >= thLB)
    constraints.append(X[N][4] <= thUB)
    constraints.append(X[N][4] >= thLB)
    constraints.append(A[N - 1] @ (X[N - 1] - s_bar[N - 1]) + B[N - 1] @ (U[N - 1] - u_bar[N - 1]) + c[N - 1] == X[N])
    objective = cvx.Minimize(cvx.sum(cost_terms))
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    s = np.zeros((N + 1, 6))
    u = np.zeros((N, 3))
    for k in range(N):
        s[k] = X[k].value
        u[k] = U[k].value
    s[N] = X[N].value
    return s, u


def dynamics(s, u):
    M = 1.  # hub mass
    m = 0.7  # boom mass
    Mm = M+m  # boom and hub mass combined
    b = 0.25  # hub side length

    l1, dl1, th1, dth1, th4, dth4 = s
    F1, M1, M4 = u

    C1 = 1
    C2 = 2 * delH1 * jnp.sin(th4) - 2 * HL * jnp.sin(th4) + 2 * l1 * jnp.sin(th1) * jnp.sin(th4) + 2 * l1 * jnp.cos(
        th1) * jnp.cos(th4) - 2 * W * jnp.cos(th4)
    C3 = jnp.square(delH1) - 2 * delH1 * HL + 2 * delH1 * l1 * jnp.sin(th1) + jnp.square(HL) - 2 * HL * l1 * jnp.sin(
        th1) + jnp.square(l1) - 2 * l1 * W * jnp.cos(th1) + jnp.square(W) - 2 * jnp.square(b)
    l4 = (-C2 - jnp.sqrt(jnp.square(C2) - 4*C1*C3))/(2*C1)

    I1 = (1/3)*m*jnp.square(l1) + M*jnp.square(l1 + b/np.sqrt(2))
    I4 = (1/3)*m*jnp.square(l4) + M*jnp.square(l4 + b/np.sqrt(2))

    ds = jnp.array([
        dl1,
        F1/Mm,
        dth1,
        -M1/I1,
        dth4,
        -M4/I4])
    return ds


if __name__ == '__main__':
    n = 6  # state dimension
    m = 3  # control dimension
    W = 3.  # hallway width
    HL = 3.  # distance between left booms in current fixed stance
    HR = 3.  # distance between right booms in current fixed stance
    delH1 = 0.  # vertical distance bottom left and bottom right booms, measured from bottom left
    start_CM = np.array([0.7, 2.7])
    goal_CM = np.array([0.5, 1.8])
    lLB, lUB, thLB, thUB = set_bounds(start_CM,goal_CM,HL,HR,delH1,W)  # boom length/angle bounds
    start_state = CM_to_state(start_CM, HL, HR, delH1, W)  # initial state vector
    goal_state = CM_to_state(goal_CM, HL, HR, delH1, W)  # desired state vector
    dt = 0.1  # discrete time resolution
    T = 30.  # total simulation time

    # specify cost function
    Qf = 100.*np.eye(6)  # terminal state cost matrix
    Q = 1.*np.eye(6)  # state cost matrix
    R = 0.1*np.eye(3)  # control cost matrix

    # specify ReachBot dynamics
    f = jax.jit(dynamics)
    f_discrete = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

    # scp parameters
    rho = 5  # trust region parameter

    # solve trajectory with scp
    print('Computing SCP solution ... ', end='')
    t = np.arange(0., T, dt)
    N = t.size - 1
    s, u = scp(f_discrete, Q, R, Qf, goal_state, start_state, N, dt, rho, lLB, lUB, thLB, thUB)
    print('done!')

    print('Simulating ...')
    for k in tqdm(range(N)):
        s[k + 1] = f_discrete(s[k], u[k])

    # visualize results
    xCMvec, yCMvec, phivec, psivec = plotReachBot(s, u, t, HL)
    #gifReachBot(xCMvec,yCMvec,phivec,psivec,W,HL,HR,delH1,s)

    # fix the s ugliness. make poly robust
