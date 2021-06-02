import numpy as np
from scipy.integrate import odeint
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cvx


@jax.partial(jax.jit, static_argnums=(0,))
def linearize(f, x, u):
    A = jax.jacfwd(f, 0)(x, u)
    B = jax.jacfwd(f, 1)(x, u)
    c = f(x, u)
    return A, B, c


def CM_to_state(CMvec, HL, HR, W):
    b = 0.25  # hub side length
    xCM, yCM = CMvec
    x_ul_corner = xCM - b/2
    y_ul_corner = yCM + b/2
    l1 = np.sqrt(np.square(0-(xCM-b/2)) + np.square(HL-(yCM+b/2)))
    th1 = np.arccos((xCM-b/2)/l1)
    l4 = np.sqrt(np.square(W-(xCM+b/2)) + np.square(0-(yCM-b/2)))
    th4 = np.arcsin((yCM-b/2)/l4)
    statevec = np.array([l1,0.,th1,0.,l4,0.,th4,0.])
    return statevec


def state_to_full(statevec):
    pass


def scp(f, Q, R, Q_N, s_star, s0, N, dt, rho, lLB, lUB, thLB, thUB):
    # Outer loop of scp
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension
    eps = 0.001  # termination threshold for scp

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
        print("State Update:", np.linalg.norm(s[N-1] - s_star))
        
    return s, u


def scp_iteration(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, lLB, lUB, thLB, thUB):
    X = {}
    U = {}
    cost_terms = []
    constraints = []
    A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)
    for k in range(N):
        X[k] = cvx.Variable(8)
        U[k] = cvx.Variable(4)
        cost_terms.append(cvx.quad_form(X[k] - s_star, Q))
        cost_terms.append(cvx.quad_form(U[k], R))
        constraints.append(X[k][0] <= lUB)
        constraints.append(X[k][0] >= lLB)
        constraints.append(X[k][2] <= thUB)
        constraints.append(X[k][2] >= thLB)
        constraints.append(X[k][4] <= lUB)
        constraints.append(X[k][4] >= lLB)
        constraints.append(X[k][6] <= thUB)
        constraints.append(X[k][6] >= thLB)
        constraints.append(cvx.norm(U[k] - u_bar[k], "inf") <= rho)
        constraints.append(cvx.norm(X[k] - s_bar[k], "inf") <= rho)
        if (k == 0):
            constraints.append(X[k] == s0)
        if (k < N and k > 0):
            constraints.append(
                A[k - 1] @ (X[k - 1] - s_bar[k - 1]) + B[k - 1] @ (U[k - 1] - u_bar[k - 1]) + c[k - 1] == X[k])
    X[N] = cvx.Variable(8)
    cost_terms.append(cvx.quad_form(X[N] - s_star, Q_N))
    constraints.append(A[N - 1] @ (X[N - 1] - s_bar[N - 1]) + B[N - 1] @ (U[N - 1] - u_bar[N - 1]) + c[N - 1] == X[N])
    objective = cvx.Minimize(cvx.sum(cost_terms))
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    s = np.zeros((N + 1, 8))
    u = np.zeros((N, 4))
    for k in range(N):
        s[k] = X[k].value
        u[k] = U[k].value
    s[N] = X[N].value

    print(prob.status)
    return s, u


def dynamics(s, u):
    M = 1.  # hub mass
    m = 0.7  # boom mass
    Mm = M+m  # boom and hub mass combined
    b = 0.25  # hub side length

    l1, dl1, th1, dth1, l4, dl4, th4, dth4 = s
    F1, M1, F4, M4 = u
    I1 = (1/3)*m*jnp.square(l1) + M*jnp.square(l1 + b/np.sqrt(2))
    I4 = (1/3)*m*jnp.square(l4) + M*jnp.square(l4 + b/np.sqrt(2))

    ds = jnp.array([
        dl1,
        F1/Mm,
        dth1,
        -M1/I1,
        dl4,
        F4/Mm,
        dth4,
        -M4/I4,
    ])
    return ds


if __name__ == '__main__':
    n = 8  # state dimension
    m = 4  # control dimension
    W = 3.  # hallway width
    HL = 3.  # distance between left booms in current fixed stance
    HR = 3.  # distance between right booms in current fixed stance
    start_CM = np.array([0.75, 0.75])
    goal_CM = np.array([2.25, 2.25])
    start_state = CM_to_state(start_CM, HL, HR, W)  # initial state
    goal_state = CM_to_state(goal_CM, HL, HR, W)  # desired state
    print("Goal: ",goal_state)
    dt = 0.1  # discrete time resolution
    T = 30.  # total simulation time

    # specify cost function
    Qf = 1000.*np.eye(8)  # terminal state cost matrix
    Q = 10.*np.eye(8)  # state cost matrix
    R = 2.5*np.eye(4)  # control cost matrix

    # specify ReachBot dynamics
    f = jax.jit(dynamics)
    f_discrete = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

    # scp parameters
    rho = 5  # trust region parameter
    lLB = 0.5  # boom length lower bound
    lUB = 4.  # boom length upper bound
    thLB = 0  # boom angle lower bound
    thUB = 1.3  # boom angle upper bound

    # solve trajectory with scp
    print('Computing SCP solution ... ', end='')
    t = np.arange(0., T, dt)
    N = t.size - 1
    s, u = scp(f_discrete, Q, R, Qf, goal_state, start_state, N, dt, rho, lLB, lUB, thLB, thUB)
    print('done!')

    print('Simulating ...')
    for k in tqdm(range(N)):
        s[k + 1] = f_discrete(s[k], u[k])
