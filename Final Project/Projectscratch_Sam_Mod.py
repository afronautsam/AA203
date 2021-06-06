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


def plotReachBot(s,u,t,HLVec,stanceVec, delHLvec, N):
    CMvec = []
    xCMvec = []
    yCMvec = []
    phivec = []
    psivec = []
    F1vec = []
    M1vec = []
    M4vec = []

    stanceNum = 0
    count = 0
    #frameshift is row 2 column 1 of each stance
    for (sk,uk) in zip(s,u):
        count += 1
        xCM, yCM, phi, psi = state_to_CM(sk,HLVec[stanceNum], delHLvec[stanceNum])
        CMvec.append(np.array([xCM,yCM]))
        xCMvec.append(xCM)
        yCMvec.append(yCM+stanceVec[2,1,stanceNum]) #visualizing solution in global frame 
        phivec.append(phi)
        psivec.append(psi)
        F1vec.append(uk[0])
        M1vec.append(uk[1])
        M4vec.append(uk[2])

        if count == N:
            count = 0 #reset count 
            stanceNum += 1 #move to next stance

    fig1 = plt.figure(1)
    plt.plot(xCMvec,yCMvec)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.title('ReachBot Optimal Trajectory')
    fig2 = plt.figure(2)
    plt.plot(t, F1vec, label='Boom 1 Force')
    plt.plot(t, M1vec, label='Boom 1 Torque')
    plt.plot(t, M4vec, label='Boom 4 Torque')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Force [N] and Torque [N.m]')
    plt.title('ReachBot Optimal Control')
    plt.grid()
    plt.legend()
    plt.show()
    return xCMvec, yCMvec, phivec, psivec


def gifReachBot(xCMvec, yCMvec, phivec, psivec, W, s, stanceVec, Nval):
    b = 0.25  # hub side length
    fig3 = plt.figure(3)
    filenames = []

    x_leftwall = np.array([0, 0])
    y_leftwall = np.array([0, 13])
    x_rightwall = np.array([W, W])
    y_rightwall = np.array([0, 13])

    count = 0
    stanceNum = 0


    for i, (xCM, yCM, phi, psi, sk) in enumerate(zip(xCMvec, yCMvec, phivec, psivec, s)):
    

        llboom = stanceVec[2,:,stanceNum]
        ulboom = stanceVec[0, :, stanceNum]
        lrboom = stanceVec[3, :, stanceNum]
        urboom = stanceVec[1,:,stanceNum]

        count += 1
        ax3 = plt.gca()
        ax3.set_axisbelow(True)
        # l4 = getl4(sk[0],sk[2],sk[4]) 
        # th4 = sk[4]
        llc = np.array([xCM-(b*np.sqrt(2)/2)*np.cos(psi), yCM-(b*np.sqrt(2)/2)*np.sin(psi)])
        ulc = np.array([xCM-(b*np.sqrt(2)/2)*np.sin(psi), yCM+(b*np.sqrt(2)/2)*np.cos(psi)])
        lrc = np.array([xCM+(b*np.sqrt(2)/2)*np.sin(psi), yCM-(b*np.sqrt(2)/2)*np.cos(psi)])
        urc = np.array([xCM+(b*np.sqrt(2)/2)*np.cos(psi), yCM+(b*np.sqrt(2)/2)*np.sin(psi)])
        #lrc = np.array([W-l4*np.cos(th4),delH1+l4*np.sin(th4)])
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

        if count == N:
            count = 0
            stanceNum += 1



    # RBot = Rectangle(xy=(xCM-(b*np.sqrt(2)/2)*np.cos(psi), yCM-(b*np.sqrt(2)/2)*np.sin(psi)), height=b, width=b, angle=(-phi*180/np.pi), edgecolor='0.4', facecolor='0.4')
    # plt.plot(x_leftwall, y_leftwall, 'k', linewidth=2)
    # plt.plot(x_rightwall, y_rightwall, 'k', linewidth=2)
    # ax3.add_patch(RBot)
    # plt.plot([llboom2[0], llc[0]], [llboom2[1], llc[1]], '0.4')
    # plt.plot([ulboom2[0], ulc[0]], [ulboom2[1], ulc[1]], '0.4')
    # plt.plot([lrboom2[0], lrc[0]], [lrboom2[1], lrc[1]], '0.4')
    # plt.plot([urboom2[0], urc[0]], [urboom2[1], urc[1]], '0.4')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.grid()
    # plt.title('ReachBot Trajectory')
    # plt.savefig(str(i+1) + '.png')
    # filenames.append(str(i+1) + '.png')
    # plt.cla()

    with imageio.get_writer('ReachBot2D.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(filenames):
        os.remove(filename)


def CM_to_state(CMvec, HLval, HRval, delH1val, W):
    b = 0.25  # hub side length
    xCM, yCM = CMvec
    l1 = np.sqrt(np.square(0 - (xCM - b / 2)) + np.square(HLval - (yCM + b / 2)))
    th1 = np.arccos((xCM - b / 2) / l1)
    l4 = np.sqrt(np.square(W - (xCM + b / 2)) + np.square(delH1val - (yCM - b / 2)))
    th4 = np.arccos((W - (xCM + b / 2)) / l4)
    statevec = np.array([l1, 0., th1, 0., th4, 0.])
    return statevec


def state_to_CM(s, HLval, delH1val):
    b = 0.25  # hub side length
    l1, dl1, th1, dth1, th4, dth4 = s
    l4 = getl4(l1,th1,th4, delH1val, HLval)  # find l4 that ensures ReachBot shape is retained
    psi = (np.arcsin((W-l1*np.cos(th1)-l4*np.cos(th4))/(b*np.sqrt(2)))).real  # corner heading angle
    phi = np.pi/4. - psi  # CM heading angle
    xCM = l1*np.cos(th1) + (b*np.sqrt(2)/2)*np.sin(psi)
    yCM = HLval - l1*np.sin(th1) - (b*np.sqrt(2)/2)*np.cos(psi)
    return xCM, yCM, phi, psi


def CM_to_full(CMvec, HLval, HRval, delH1val, W):
    b = 0.25  # hub side length
    xCM, yCM = CMvec
    l1 = np.sqrt(np.square(0 - (xCM - b / 2)) + np.square(HLval - (yCM + b / 2)))
    th1 = np.arccos((xCM - b / 2) / l1)
    l2 = np.sqrt(np.square(W - (xCM + b / 2)) + np.square(HRval - (yCM - delH1val + b / 2)))
    th2 = np.arccos((W - (xCM + b / 2)) / l2)
    l3 = np.sqrt(np.square(0 - (xCM - b / 2)) + np.square(0 - (yCM - b / 2)))
    th3 = np.arccos((xCM - b / 2) / l3)
    l4 = np.sqrt(np.square(W - (xCM + b / 2)) + np.square(delH1val - (yCM - b / 2)))
    th4 = np.arccos((W - (xCM + b / 2)) / l4)
    full_lvec = np.array([l1, l2, l3, l4])
    full_thvec = np.array([th1, th2, th3, th4])
    return full_lvec, full_thvec


def getl4(l1,th1,th4, delH1val, HLval):
    b = 0.25  # hub side length
    C1 = 1
    C2 = 2*delH1val*np.sin(th4) - 2*HLval*np.sin(th4) + 2*l1*np.sin(th1)*np.sin(th4) + 2*l1*np.cos(th1)*np.cos(th4) - 2*W*np.cos(th4)
    C3 = np.square(delH1val) - 2*delH1val*HLval + 2*delH1val*l1*np.sin(th1) + np.square(HLval) - 2*HLval*l1*np.sin(th1) + np.square(l1) - 2*l1*W*np.cos(th1) + np.square(W) - 2*np.square(b)
    poly = np.array([C1, C2, C3])
    lengths = np.roots(poly)
    if delH1val > 0:
        delH1val = -1*delH1val

    for lposs in lengths:
        if lposs.real*np.sin(th4) + l1*np.sin(th1) < (HLval-delH1val + 0.1):
            l4 = lposs
    return l4
    


def set_bounds(CM1, CM2, HLval, HRval, delH1val, W):
    # establish reasonable extrema
    llb = 0.5
    lub = 4.
    thlb = -np.pi/6
    thub = 75.*np.pi/180.

    # find start/goal extrema
    full_lvec1, full_thvec1 = CM_to_full(CM1, HLval, HRval, delH1val, W)
    full_lvec2, full_thvec2 = CM_to_full(CM2, HLval, HRval, delH1val, W)
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
    constraints.append(X[N] == s_star)
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

def wayPoints(currStance, nextStance):
    HL = currStance[0,1] - currStance[2,1]
    HR = currStance[1,1] - currStance[3,1]
    delH1 = -currStance[2,1] + currStance[3,1]

    # HL_next = nextStance[0,1] - nextStance[2,1]
    # HR_next = nextStance[1,1] - nextStance[3,1]
    # delH1_next = -nextStance[2,1] + nextStance[3,1]

    change = nextStance - currStance
    changedInd = np.argwhere(change > 0)

    dist = nextStance[changedInd[0,0],1] - nextStance[:,1]
    distMod = np.where(dist > 0, dist, 100)
    neighborInd = np.argmin(distMod)


    

    if changedInd[0,0] < 2:
        goalYCoord = currStance[neighborInd,1] - 1 #go up or down depending on stance change
        #goalYCoord = currStance[2,1] + 2
        goalXCoord = np.abs(2.5 - nextStance[changedInd[0,0], 0]) #shift to the left or right depending on stance 
    else:
        goalYCoord = currStance[neighborInd, 1] + 2
        goalXCoord = 1.5 #move to the middle when moving bottom legs
        #goalYCoord = currStance[2,1] + 2
    
    
    # if np.abs(currStance[2,1] - goalYCoord) > 3:
    #     goalYCoord = currStance[2,1] + 2
    goalCM = np.array([goalXCoord, goalYCoord])
    return goalCM, HL, HR, delH1


if __name__ == '__main__':
    n = 6  # state dimension
    m = 3  # control dimension
    W = 3.  # hallway width


    
    stance0 = np.array([[0, 5.2], [W, 4.9], [0, 1.7], [W, 0.8]]) #sample stance in a global frame
    stance1 = np.array([[0, 5.2], [W, 7.8], [0, 1.7], [W, 0.8]]) #next stance in a global frame
    stance2 = np.array([[0, 5.2], [W, 7.8], [0, 1.7], [W, 4.9]])
    stance3 = np.array([[0, 9.2], [W, 7.8], [0, 1.7], [W, 4.9]])
    stance4 = np.array([[0, 9.2], [W, 7.8], [0, 5.2], [W, 4.9]])
    stance5 = np.array([[0, 9.2], [W, 11.1], [0, 5.2], [W, 4.9]])
    stance6 = np.array([[0, 9.2], [W, 11.1], [0, 5.2], [W, 7.8]])
    stance7 = np.array([[0, 13.2], [W, 11.1], [0, 5.2], [W, 7.8]])
    stance8 = np.array([[0, 13.2], [W, 11.1], [0, 9.2], [W, 7.8]])


    stanceVec = np.dstack([stance0, stance1, stance2, stance3, stance4, stance5, stance6, stance7, stance8])


    # currStance = np.array([[0, 5.2], [W, 4.9], [0, 1.7], [W, 0.8]]) #sample stance in a global frame
    # midStance = np.array([[0, 5.2], [W, 7.8], [0, 1.7], [W, 0.8]]) #next stance in a global frame 
    # nextStance = np.array([[0, 5.2], [W, 7.8], [0, 1.7], [W, 4.9]])

    start_CM = np.array([2.0, 2.5]) #specified in a global frame
    goal_CM = np.array([2.5, 11.7])

    CM1, HL1, HR1, delH11 = wayPoints(stance0, stance1)
    CM2, HL2, HR2, delH12 = wayPoints(stance1, stance2)
    CM3, HL3, HR3, delH13 = wayPoints(stance2, stance3)
    CM4, HL4, HR4, delH14 = wayPoints(stance3, stance4)
    CM5, HL5, HR5, delH15 = wayPoints(stance4, stance5)
    CM6, HL6, HR6, delH16 = wayPoints(stance5, stance6)
    CM7, HL7, HR7, delH17 = wayPoints(stance6, stance7)
    CM8, HL8, HR8, delH18 = wayPoints(stance7, stance8)
    
    
    start_CM_shift = start_CM - stance0[2,:] #to be in the frame of the solver
    goal_CM_shift = goal_CM - stance8[2,:]
    CM1_shift = CM1 - stance0[2,:]
    CM2_shift = CM2 - stance1[2,:] #has to be between 0 and 3
    CM3_shift = CM3 - stance2[2,:]
    CM4_shift = CM4 - stance3[2,:]
    CM5_shift = CM5 - stance4[2,:]
    CM6_shift = CM6 - stance5[2,:]
    CM7_shift = CM7 - stance6[2,:]
    #CM7_shift[1] = CM7_shift[1]-2.7
    CM8_shift = CM8 - stance7[2,:]

    delH17 = 3 #a bit of a hack to get it to run
    delH18 = -0.5
    delHf = -0.5
    Hf = 3.7



    lLB1, lUB1, thLB1, thUB1 = set_bounds(start_CM_shift,CM1_shift,HL1,HR1,delH11,W)  # boom length/angle bounds
    lLB2, lUB2, thLB2, thUB2 = set_bounds(CM1_shift,CM2_shift,HL2,HR2,delH12,W)  # boom length/angle bounds
    lLB3, lUB3, thLB3, thUB3 = set_bounds(CM2_shift,CM3_shift,HL3,HR3,delH13,W)  # boom length/angle bounds
    lLB4, lUB4, thLB4, thUB4 = set_bounds(CM3_shift,CM4_shift,HL4,HR4,delH14,W)  # boom length/angle bounds
    lLB5, lUB5, thLB5, thUB5 = set_bounds(CM4_shift,CM5_shift,HL5,HR5,delH15,W)  # boom length/angle bounds
    lLB6, lUB6, thLB6, thUB6 = set_bounds(CM5_shift,CM6_shift,HL6,HR6,delH16,W)  # boom length/angle bounds
    lLB7, lUB7, thLB7, thUB7 = set_bounds(CM6_shift,CM7_shift,HL7,HR7,delH17,W)  # boom length/angle bounds #revert
    lLB8, lUB8, thLB8, thUB8 = set_bounds(CM7_shift,CM8_shift,HL8,HR8,delH18,W)  # boom length/angle bounds
    lLB9, lUB9, thLB9, thUB9 = set_bounds(CM8_shift,goal_CM_shift,Hf,HL8-1,delHf,W)  # boom length/angle bounds

    start_state = CM_to_state(start_CM_shift, HL1, HR1, delH11, W)  # initial state vector
    state1 = CM_to_state(CM1_shift, HL1, HR1, delH11, W)  # desired state vector
    state2 = CM_to_state(CM2_shift, HL2, HR2, delH12, W)  # desired state vector
    state3 = CM_to_state(CM3_shift, HL3, HR3, delH13, W)  # desired state vector
    state4 = CM_to_state(CM4_shift, HL4, HR4, delH14, W)  # desired state vector
    state5 = CM_to_state(CM5_shift, HL5, HR5, delH15, W)  # desired state vector
    state6 = CM_to_state(CM6_shift, HL6, HR6, delH16, W)  # desired state vector
    state7 = CM_to_state(CM7_shift, HL7, HR7, delH17, W)  # desired state vector
    state8 = CM_to_state(CM8_shift, HL8, HR8, delH18, W)  # desired state vector
    goal_state = CM_to_state(goal_CM_shift, Hf, HL8-1, delHf, W)  # desired state vector

    # lVec, thetVec = CM_to_full(goal_CM_shift, HL, HR, delH1, W)

    # if np.any(lVec) > 4:
    #     raise ValueError("Requires violating control force constraints")
    # elif np.any(thetVec) > 75.*np.pi/180: 
    #     raise ValueError("Requires violating control torque constraints")

    dt = 0.1  # discrete time resolution
    T = 30.  # total simulation time

    # specify cost function
    Qf = 100.*np.eye(6)  # terminal state cost matrix
    Q = 1.*np.eye(6)  # state cost matrix
    R = 0.1*np.eye(3)  # control cost matrix
    #R[0,0] = 1 #imposing a lower cost on linear control
    # specify ReachBot dynamics
    f = jax.jit(dynamics)
    f_discrete = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

    # scp parameters  # desired s
    rho = 5  # trust region parameter

    # solve trajectory with scp and simulate with discrete time dynamics
    print('Computing SCP solution ... ', end='')
    t = np.arange(0., T, dt)
    N = t.size - 1
    delH1 = delH11
    HL = HL1
    s1, u1 = scp(f_discrete, Q, R, Qf, state1, start_state, N, dt, rho, lLB1, lUB1, thLB1, thUB1)
    print('Done with Problem 1')
    print('Simulating ...')
    for k in tqdm(range(N)):
        s1[k + 1] = f_discrete(s1[k], u1[k])
    delH1 = delH12
    HL = HL2
    s2, u2 = scp(f_discrete, Q, R, Qf, state2, state1, N, dt, rho, lLB2, lUB2, thLB2, thUB2)
    print('Done with Problem 2')
    print('Simulating ...')
    for k in tqdm(range(N)):
        s2[k + 1] = f_discrete(s2[k], u2[k])
    delH1 = delH13
    HL = HL3
    s3, u3 = scp(f_discrete, Q, R, Qf, state3, state2, N, dt, rho, lLB3, lUB3, thLB3, thUB3)
    print('Done with Problem 3')
    print('Simulating ...')
    for k in tqdm(range(N)):
        s3[k + 1] = f_discrete(s3[k], u3[k])
    delH1 = delH14
    HL = HL4
    s4, u4 = scp(f_discrete, Q, R, Qf, state4, state3, N, dt, rho, lLB4, lUB4, thLB4, thUB4)
    print('Done with Problem 4')
    print('Simulating ...')
    for k in tqdm(range(N)):
        s4[k + 1] = f_discrete(s4[k], u4[k])
    delH1 = delH15
    HL = HL5
    s5, u5 = scp(f_discrete, Q, R, Qf, state5, state4, N, dt, rho, lLB5, lUB5, thLB5, thUB5)
    print('Done with Problem 5')
    print('Simulating ...')
    for k in tqdm(range(N)):
        s5[k + 1] = f_discrete(s5[k], u5[k])
    delH1 = delH16
    HL = HL6
    s6, u6 = scp(f_discrete, Q, R, Qf, state6, state5, N, dt, rho, lLB6, lUB6, thLB6, thUB6)
    print('Done with Problem 6')
    print('Simulating ...')
    for k in tqdm(range(N)):
        s6[k + 1] = f_discrete(s6[k], u6[k])
    delH1 = delH17
    HL = HL7
    s7, u7 = scp(f_discrete, Q, R, Qf, state7, state6, N, dt, rho, lLB7, lUB7, thLB7, thUB7)
    print('Done with Problem 7')
    print('Simulating ...')
    for k in tqdm(range(N)):
        s7[k + 1] = f_discrete(s7[k], u7[k])
    # delH1 = delH18
    # HL = HL8
    # s8, u8 = scp(f_discrete, Q, R, Qf, state8, state7, N, dt, rho, lLB8, lUB8, thLB8, thUB8)
    # print('Done with Problem 8')
    # print('Simulating ...')
    # for k in tqdm(range(N)):
    #     s8[k + 1] = f_discrete(s8[k], u8[k])
    # s9, u9 = scp(f_discrete, Q, R, Qf, goal_state, state8, N, dt, rho, lLB9, lUB9, thLB9, thUB9)
    # print('Done with Planning Problem')
    # print('Simulating ...')
    # for k in tqdm(range(N)):
    #     s9[k + 1] = f_discrete(s9[k], u9[k])


   
    tAll = np.arange(0., 7*T -7*dt, dt)
    t = np.arange(0., T - dt, dt)

    HLvec = np.vstack([HL1, HL2, HL3, HL4, HL5, HL6, HL7, HL8,HL8])
    delHLvec = np.vstack([delH11, delH12, delH13, delH14, delH15, delH16, delH17, delH18, delH18])

    # print('Simulating ...')
    # for k in tqdm(range(N)):
    #     s1[k + 1] = f_discrete(s1[k], u1[k])
    #     s2[k + 1] = f_discrete(s2[k], u2[k])
    #     s3[kH + 1] = f_discrete(s3[k], u3[k])


    sComb = np.vstack([s1,s2,s3,s4,s5,s6,s7])
    uComb = np.vstack([u1,u2,u3,u4,u5,u6,u7])
    # visualize results
    xCMvec, yCMvec, phivec, psivec = plotReachBot(sComb, uComb, tAll, HLvec, stanceVec, delHLvec, N)
    gifReachBot(xCMvec,yCMvec,phivec,psivec,W,sComb, stanceVec,N)

    # xCMvec, yCMvec, phivec, psivec = plotReachBot(s1, u1, t, HLvec, stanceVec, delHLvec, N)
    # gifReachBot(xCMvec,yCMvec,phivec,psivec,W,s1, stanceVec,N)

    # fix the s ugliness. make poly robust
