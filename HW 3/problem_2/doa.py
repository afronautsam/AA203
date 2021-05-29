import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cvx
from numpy.linalg import inv
from numpy import linalg as LA


def convCheck(Q,R,P,s0,sf, N,uB,sB):
    #run for all time steps
    u0 = 0
    NMax = N
    feasflag = 0
    s = np.zeros((1,2)) #initialize s 
    u = np.zeros((1,1)) #initialize u
    sCapture = np.zeros((1,N+1,2)) #initialize sCapture
    A = np.array(([1,1],[0,1]))
    B = np.array(([0], [1]))

    i = 0
    print("s0:", s0)

    while (np.linalg.norm(s0[0]) > sf[0] or np.linalg.norm(s0[1]) > sf[0]):

        if N < 1:
            break

        sStep,uStep = convex_problem(Q,R,P,s0,u0,N,uB,sB,NMax) #solve the full convex problem at each timestep

        if np.any(uStep == None): #check feasibility
            print("Infeasible!")
            feasflag = 0
            break
        else:
            feasflag = 1

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
        #N -= 1
        i += 1
        


    
    #return feasflag
    return s,u,sCapture

def doa(Q,R,P,sf, N,uB,sB):
    stateGrid = np.linspace(-sB,sB,51)
    attracList = np.zeros((1,2))
    pointList = np.zeros((1,2))
    feascount = 1

    for ii in range(len(stateGrid)):
        for jj in range(len(stateGrid)):
            s0 = np.array([stateGrid[ii],stateGrid[jj]])
            feasflag = convCheck(Q,R,P,s0,sf,N,uB,sB)

            if feasflag == 1:
                if feascount == 0:
                    attracList = s0
                else:
                   attracList = np.vstack((attracList,s0))
                feascount += 1
        
            if ii == 6:
                kroo = 1 + 2
            if ii == 0 and jj == 0:
                pointList = s0
            else:
                pointList = np.vstack((pointList, s0))

    return attracList,pointList




def convex_problem(Q,R,P,s0,u0,N,uB, xB, NMax):
    cost_terms = []
    constraints = []
    s =[]
    u = []

    s = cvx.Variable((N+1, 2))
    u = cvx.Variable((N, 1))

    A = np.array(([1,1],[0,1]))
    B = np.array(([0], [1]))

    for t in range(N+1):
        if t < N:
            cost_terms.append(0.5*cvx.quad_form((s[t]), Q)) #defining cost terms 
            cost_terms.append(0.5*cvx.quad_form(u[t], R))
        else:
            cost_terms.append(0.5*cvx.quad_form((s[t]), P)) #terminal cost 
        
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
        else:
            constraints.append(s[t] == 0) #terminal constraint



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
    sf = np.array([1e-6, 1e-6])
    s0 = np.array([-5.0, 2.5])

    A = np.array(([1,1],[0,1]))
    B = np.array(([0], [1]))

    initP = np.zeros((2,2))


    # specify cost function

    Q = 1*np.eye(2)               # state cost matrix
    R = 0.01*np.eye(1)                  # control cost matrix

    #constraints
    uB = 1                 # control effort lower bound
    sB = 10.                    # control effort upper bound

    P = Riccati(R,A,B,Q,initP)                   # terminal state cost matrix
 
    N1 = 2
    N2 = 4
    N3 = 6
    N4 = 8
    N5 = 10

    # doa1,points = doa(Q,R,P,sf,N,uB,sB)
    # fig, ax = plt.subplots()

    
    # ax.plot(points[:,0], points[:,1],'ro')
    # ax.plot(doa1[:,0], doa1[:,1],'bo')

    s1,u1,sCapture1 = convCheck(Q,R,P,s0,sf,N1,uB,sB)
    s2,u2,sCapture2 = convCheck(Q,R,P,s0,sf,N2,uB,sB)
    s3,u3,sCapture3 = convCheck(Q,R,P,s0,sf,N3,uB,sB)
    s4,u4,sCapture4 = convCheck(Q,R,P,s0,sf,N4,uB,sB)
    s5,u5,sCapture5 = convCheck(Q,R,P,s0,sf,N5,uB,sB)

    fig, ax = plt.subplots()
    ax.grid()
    # ax.plot(s1[:,0], s1[:,1])
    # ax.plot(s1[:,0], s1[:,1], 'bo')

    
    # for i in range(sCapture1.shape[0]):
    #     ax.plot(sCapture1[i,:,0], sCapture1[0,:,1], 'k--')
    #     ax.plot(sCapture1[i,:,0], sCapture1[0,:,1],'ko')
    
    # for i in range(sCapture2.shape[0]):
    #     ax.plot(sCapture2[i,:,0], sCapture2[0,:,1], 'r--')
    #     ax.plot(sCapture2[i,:,0], sCapture2[0,:,1],'ro')
    
    # for i in range(sCapture3.shape[0]):
    #     ax.plot(sCapture3[i,:,0], sCapture3[0,:,1], 'b--')
    #     ax.plot(sCapture3[i,:,0], sCapture3[0,:,1],'bo')

    # for i in range(sCapture4.shape[0]):
    #     ax.plot(sCapture4[i,:,0], sCapture4[0,:,1], 'g--')
    #     ax.plot(sCapture4[i,:,0], sCapture4[0,:,1],'go')

    # for i in range(sCapture5.shape[0]):
    #     ax.plot(sCapture5[i,:,0], sCapture5[0,:,1], 'y--')
    #     ax.plot(sCapture5[i,:,0], sCapture5[0,:,1],'yo')

    ax.plot(s2[:,0], s2[:,1], 'r')
    ax.plot(s2[:,0], s2[:,1], 'ro')
    ax.plot(s3[:,0], s3[:,1], 'b')
    ax.plot(s3[:,0], s3[:,1], 'bo')
    ax.plot(s4[:,0], s4[:,1], 'g')
    ax.plot(s4[:,0], s4[:,1], 'go')
    ax.plot(s5[:,0], s5[:,1], 'y')
    ax.plot(s5[:,0], s5[:,1], 'yo')
    

    plt.show()
    print('done!')

   
