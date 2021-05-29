import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cvx


def MPC(Q,R,P,s0,sf, N,uB,sB):
    #run for all time steps
    u0 = 0
    NMax = N
    s = np.zeros((1,2)) #initialize s 
    u = np.zeros((1,1)) #initialize u
    sCapture = np.zeros((1,N+1,2)) #initialize sCapture
    A = np.array(([1,1],[0,1]))
    B = np.array(([0], [1]))
    eps = 1e-10
    i = 0

    while (np.linalg.norm(s0[0]) > sf[0] and np.linalg.norm(s0[1]) > sf[0]):
        sStep,uStep = convex_problem(Q,R,P,s0,u0,N,uB,sB,NMax) #solve the full convex problem at each timestep

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


if __name__ == '__main__':
    s0 = np.array([-4.5,2])      
    sf = np.array([1e-10, 1e-10])

    # specify cost function
    P = 1.*np.eye(2)                    # terminal state cost matrix
    Q = 1*np.eye(2)                     # state cost matrix
    R = 10*np.eye(1)                  # control cost matrix

    #constraints
    uB = 0.5                 # control effort lower bound
    sB = 5.                    # control effort upper bound

    # solve swing-up with scp
    print('Computing MPC solution ... ', end='')
             
    N = 3                         
    s1,u1,sCapture1 = MPC(Q,R,P,s0,sf,N,uB,sB)
    print('done!')

    s0 = np.array([-4.5, 3])
    
    s2,u2,sCapture2 = MPC(Q,R,P,s0,sf,N,uB,sB)

    print('Simulating ...')

    # Plot
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(s1[:,0], s1[:,1])
    ax.plot(s1[:,0], s1[:,1], 'bo')
    ax.plot(sCapture1[0,:,0], sCapture1[0,:,1], 'k--')
    ax.plot(sCapture1[1,:,0], sCapture1[1,:,1], 'k--')
    ax.plot(sCapture1[2,:,0], sCapture1[2,:,1], 'k--')
    ax.plot(sCapture1[0,:,0], sCapture1[0,:,1],'ko')
    ax.plot(sCapture1[1,:,0], sCapture1[1,:,1],'ko')
    ax.plot(sCapture1[2,:,0], sCapture1[2,:,1],'ko')

    ax.plot(s2[:,0], s2[:,1])
    ax.plot(s2[:,0], s2[:,1],'bo')
    ax.plot(sCapture2[0,:,0], sCapture2[0,:,1], 'k--')
    ax.plot(sCapture2[1,:,0], sCapture2[1,:,1], 'k--')
    ax.plot(sCapture2[0,:,0], sCapture2[0,:,1], 'ko')
    ax.plot(sCapture2[1,:,0], sCapture2[1,:,1], 'ko')

    

    plt.show()

