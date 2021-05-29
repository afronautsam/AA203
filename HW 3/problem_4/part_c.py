from model import dynamics, cost
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt 

stochastic_dynamics = True # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()
gamma = 1

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


Aknown = dynfun.A
Bknown = dynfun.B
Qknown = costfun.Q
Rknown = costfun.R

eps = 1e-6

L_star,P_star = Riccati(Aknown,Bknown,Qknown,Rknown,eps)


T = 100
N = 100

total_costs = []

#intialize the value function
np.random.seed(0)
Q = np.random.rand(6,6) #maybe use a seed?
Q = 0.5*(Q + Q.T) #make symmetric 

pQ0 = 1e8*np.eye(36) #trying to learn 36 parameters 
Q22 = Q[4:,4:] #a P.D 2x2
Q21 = Q[4:,0:4]

L = np.zeros((2,4))
timeCount = 0

cQ0 = Q.ravel().reshape(36,1) #check shape should be 36x1 c matrix

for n in range(N):
    costs = []
    
    pQ0 = 1e8*np.eye(36) #trying to learn 36 parameters 

    x = dynfun.reset()
    for t in range(T):

        # TODO compute action
        u = -L@x + np.random.normal(0,1,size=(2))

        xu = np.concatenate((x,u))
        xu_bar = np.outer(xu,xu).ravel() #36x1 z matrix

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)

        # TODO recursive least squares polict evaluation step
        pQ = pQ0 - ((pQ0 @ np.outer(xu_bar, xu_bar) @ pQ0)/(1 + xu_bar.T @ pQ0 @ xu_bar))
        
        cQ = cQ0 + np.outer(pQ0 @ xu_bar,c - cQ0.T @ xu_bar)/(1 + xu_bar.T @ pQ0 @ xu_bar) #should return a 36x1

        pQ0 = np.copy(pQ)
        cQ0 = np.copy(cQ)


        x = xp.copy()

    
    # TODO policy improvement step
    Q = cQ.reshape(6,6)
    Q22 = Q[4:,4:]
    Q21 = Q[4:,0:4]
    L = -LA.inv(Q22) @ Q21
    
    total_costs.append(sum(costs))

    if n == 0:
        LnormVal = LA.norm(L_star + L)
        normTime = np.array([timeCount])
        costHold = np.array([sum(costs)])
        timeCount += 1
    else:
        LnormVal = np.vstack((LnormVal, (LA.norm(L_star + L))))
        timeCount += 1
        normTime = np.vstack((normTime, np.array([timeCount])))
        costHold = np.vstack((costHold, np.array([sum(costs)])))

episodes = np.arange(N)
fig1, ax1 = plt.subplots()
ax1.plot(normTime, LnormVal)

fig2, ax2 = plt.subplots()
ax2.plot(episodes, costHold)

plt.show()

poo = 1+2