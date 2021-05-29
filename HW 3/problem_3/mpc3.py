import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import cvxpy as cvx
from numpy.linalg import inv
from numpy import linalg as LA


def MPC(Q,M,R,P,s0,sf, N,uB,sB):
    #run for all time steps
    u0 = 0
    NMax = N
    s = np.zeros((1,2)) #initialize s 
    u = np.zeros((1,1)) #initialize u
    sCapture = np.zeros((1,N+1,2)) #initialize sCapture
    A = np.array(([0.95,0.5],[0,0.95]))
    B = np.array(([0], [1]))
    i = 0

    while (np.linalg.norm(s0[0]) > 0.01 or np.linalg.norm(s0[1]) > 0.01):
        sStep,uStep = convex_problem(Q,M,R,P,s0,sf,u0,N,uB,sB,NMax) #solve the full convex problem at each timestep

        if np.any(uStep == None): #check feasibility
            print("Infeasible!")
            break

        if i == 0:
            s = s0 #initial condition
            u = uStep[0] #initial optimal control
            sCapture[i] = sStep #first layer
        else:
            sStep3d = np.expand_dims(sStep, axis=0) #to make it possible to concatenate
            sCapture = np.concatenate((sCapture,sStep3d)) #add a new layer
            u = np.vstack((u, uStep[0])) #define the action using the optimal action of the convex problem

        sNew = A@sStep[0] + B @uStep[0] #define the state using the solution of the convex problem
        s = np.vstack((s,sNew))
        

        s0 = s[i+1]
        i += 1
        


    
    return s,u,sCapture

def convex_problem(Q,M,R,P,s0,sf,u0,N,uB, xB, NMax):
    cost_terms = []
    constraints = []
    s =[]
    u = []

    s = cvx.Variable((N+1, 2))
    u = cvx.Variable((N, 1))

    A = np.array(([0.95,0.5],[0,0.95]))
    B = np.array(([0], [1]))

    for t in range(N+1):
            if t < N:
                cost_terms.append(0.5*cvx.quad_form((s[t]), Q)) #defining cost terms 
                cost_terms.append(0.5*cvx.quad_form(u[t], R))
            # else:
            #     cost_terms.append(0.5*cvx.quad_form((s[t]), P)) #terminal cost 
            
            if t == 0:
                constraints.append(s[t] == s0)

            if N != NMax and t == 0: #excluding the first timestep
                constraints.append(u[0] == u0) #use the optimal control from the previous timestep in this step

            if t < N:
                constraints.append(u[t] >= -uB) #lower bound control constraint 
                constraints.append(u[t] <= uB) #upper bound control constraint 
            
                
            if t < N:
                constraints.append(s[t+1] == A @ s[t] + B @ u[t]) #dynamics constraint 
                constraints.append(s[t] >= -xB) #lower bound state constraint 
                constraints.append(s[t] <= xB) #upper bound state constraint  
            # else:
            #     constraints.append(cvx.quad_form(s[t],M) <= 1)





    costSum = cvx.sum(cost_terms)
    objective = cvx.Minimize(costSum)
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    print(prob.status)
    print(prob.value)

    s = s.value
    u = u.value

    ###########################################################################
    return s,u


def Riccati(R, A, B, Q, PkOld):
    Kk = -inv(R + B.T@PkOld@B)@B.T@PkOld@A
    Pk = Q + A.T@PkOld@(A + B@Kk)

    while LA.norm(Pk - PkOld) > 1e-4:
        PkOld = Pk
        Kk = -inv(R + B.T@PkOld@B)@B.T@PkOld@A
        Pk = Q + A.T@PkOld@(A + B@Kk)

    return Pk


if __name__ == '__main__':
    sf = np.array([5, 1])
    s0 = np.array([-3.0, -2.5])

    A = np.array(([0.95,0.5],[0,0.95]))
    B = np.array(([0], [1]))

    M = np.array(([0.04,0],[0,1.06]))
    #constraints
    uB = 1.                 # control effort lower bound
    sB = 5.                    # control effort upper bound

    Q = 1*np.eye(2)                     # state cost matrix
    R = 1*np.eye(1)                  # control cost matrix

    initP = np.zeros((2,2))
    P = Riccati(R,A,B,Q,initP)                   # terminal state cost matrix

    # solve swing-up with scp
    print('Computing MPC solution ... ', end='')
             
    N = 4                         
    s1,u1,sCapture1 = MPC(Q,M,R,P,s0,sf,N,uB,sB)
    fig, ax = plt.subplots()
    ax.grid()

    ell1 = patch.Ellipse((0.,0.), 5, 1, color='green', fill=False)

    circ1 = plt.Circle((0,0), 5, color='grey', fill=False)
    ax.add_patch(circ1)
    ax.add_patch(ell1)
  


    for i in range(sCapture1.shape[0]):
        #ax.plot(sCapture1[i], 'k--')
        ax.plot(sCapture1[i,:,0], sCapture1[i,:,1], 'k--')
        ax.plot(sCapture1[i,:,0], sCapture1[i,:,1],'ko')
    
    ax.plot(s1[:,0], s1[:,1], 'r')
    ax.plot(s1[:,0], s1[:,1], 'bo')
    ax.set_title("Trajectory Taken, Planned Trajectories, Invariant Set Constraint and State Constraint")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    fig2, ax2 = plt.subplots()
    ax2.plot(u1)
    ax2.set_title("Trajectory Taken, Planned Trajectories, Invariant Set Constraint and State Constraint")
    ax2.set_xlabel("Number of Iterations")
    ax2.set_ylabel("Optimal Control Action")

    
    plt.show()
    print('done!')


