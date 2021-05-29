from numpy.core.shape_base import block
from model import dynamics, cost
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import block_diag
import matplotlib.pyplot as plt 




stochastic_dynamics = True # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

def Riccati(A,B,Q,R,eps):
    PkOld = np.zeros((len(A), len(A)))
    ricCount = 0

    L = -LA.inv(R + B.T@PkOld@B)@B.T@PkOld@A
    Pk = gamma*(Q + A.T@PkOld@(A + B@L))

    #while LA.norm(Pk - PkOld) > eps:
    while (not np.allclose(Pk, PkOld)) and (ricCount < 5000):
        print(LA.norm(Pk - PkOld))
        PkOld = Pk
        L = -LA.inv(R + B.T@PkOld@B)@B.T@PkOld@A
        Pk = gamma*(Q + A.T@PkOld@(A + B@L))
        ricCount += 1

    print("done!")
    return L,Pk


total_costs = []

Aknown = dynfun.A
Bknown = dynfun.B
Qknown = costfun.Q
Rknown = costfun.R

eps = 1e-6

L_star,P_star = Riccati(Aknown,Bknown,Qknown,Rknown,eps)

timeCount = 0

np.random.seed(0)
A = np.random.rand(4,4) #maybe use a seed?
B = np.random.rand(4,2)
Q = np.eye(4) #defining state cost
R = np.eye(2) #defining control cost 


cDyn0 = np.vstack((A.T,B.T)) #make psd? 6x4
cCost0 = np.zeros((20,1)) #20x1


pC0 = np.eye(20) #initialize P for cost regression
pX0 = np.eye(6) #initialize P for dynamics regression
timeCount = 0

for n in range(N):
    costs = []
    
    x = dynfun.reset() #new initial x for next training episode

    #qSmall = Q.reshape(16,1)
    qSmall = Q.ravel() #review
    #rSmall = R.reshape(4,1)
    rSmall = R.ravel()

    cCost0 = np.concatenate((qSmall, rSmall)).reshape((20,1)) #wrap it into a 20x1

  
    for t in range(T): #run the episode 

        # for i in range(len(x)): #put x in form requested by cost problem
        #     for j in range(len(x)):
        #         if i == 0 and j == 0:
        #             xr = x[i]*x[j]
        #         else:
        #             xComp = x[i]*x[j]
        #             xr = np.vstack((xr, xComp))

        xr = np.outer(x,x).ravel()
        

        # TODO compute policy
        eps = 1e-3
        L,P = Riccati(A,B,Q,R,eps)
        
        # TODO compute action
        u = L@x
        count = 0

        # for i in range(len(u)): #put u in form requested by cost problem
        #     for j in range(len(u)):
        #         if i == 0 and j == 0:
        #             up = u[i]*u[j]
        #         else:
        #             uComp = u[i]*u[j]
        #             up = np.vstack((up, uComp))

        up = np.outer(u,u).ravel()
        
        # get reward
        zC = np.concatenate((xr,up)) #20x0
        c = costfun.evaluate(x,u)
        print("Cost at timestep: ", c)
        costs.append((gamma**t)*c) 

        pC = pC0 - ((pC0 @ np.outer(zC, zC) @ pC0)/(1 + zC.T @ pC0 @ zC))
        
        cCost = cCost0 + np.outer(pC0 @ zC,c - cCost0.T @ zC)/(1 + zC.T @ pC0 @ zC) #should return a 20x1

        pC0 = np.copy(pC) #update
        cCost0 = np.copy(cCost) #update 


        # print("Old Q:", Q)
        # print("Old R:", R)


        qSmallNew = cCost[0:16]
        rSmallNew = cCost[16:20]

        Q = qSmallNew.reshape(4,4)
        R = rSmallNew.reshape(2,2)

        # print("New Q:", Q)
        # print("New R:", R) 

        #updated Q and R
        #now update dynamics
    
        # dynamics step
        xp = dynfun.step(u) #x_t+1

        
        #xp = np.reshape(xp, (1,-1))
        # x = np.reshape(x, (-1,1)) #make 2d
        # u = np.reshape(u, (-1,1)) #ditto

        zX = np.concatenate((x,u)) #a 6x1

         
        #C.T is a 4x6 for dynamics regression so CDyn is a 6x4
        #CDyn0 is a 6x4
        pX = pX0 - (pX0 @ np.outer(zX,zX) @ pX0)/(1 + zX.T @ pX0 @ zX) #6x6
        cDyn = cDyn0 + np.outer(pX0 @ zX,xp -cDyn0.T @ zX)/(1 + zX.T @ pX0 @zX) #should return a 6x4

        cDyn0 = np.copy(cDyn) #cDyn0 has to be 4x6
        pX0 = np.copy(pX) #update pX0

        # print("Old A:", A)
        # print("Old B:", B) 

        #Update A and B
        A = cDyn[:4,:].T
        B = cDyn[4:,:].T #has to be 4x2

        # print("New A:", A)
        # print("New B:", B) 



        x = xp.copy() #update x
    

        if t == 0 and n == 0:
            LnormVal = LA.norm(L_star - L)
            normTime = np.array([timeCount])
            timeCount += 1
        else:
            LnormVal = np.vstack((LnormVal, (LA.norm(L_star - L))))
            timeCount += 1
            normTime = np.vstack((normTime, np.array([timeCount])))


        
    total_costs.append(sum(costs))
    print("Total Cost: ", sum(costs))

    if n == 0:
        costHold = np.array([sum(costs)])
    else:
        costHold = np.vstack((costHold, np.array([sum(costs)])))

episodes = np.arange(N)        

fig1, ax1 = plt.subplots()
ax1.plot(normTime, LnormVal)

fig2, ax2 = plt.subplots()
ax2.plot(episodes, costHold)

plt.show()