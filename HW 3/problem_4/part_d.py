from model import dynamics, cost
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt 

stochastic_dynamics = True # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100
N = 10000 
gamma = 0.95 # discount factor

total_costs = []

def Riccati(A,B,Q,R,eps):
    PkOld = np.zeros((len(A), len(A)))
    ricCount = 0

    L = -LA.inv(R + B.T@PkOld@B)@B.T@PkOld@A
    Pk = gamma*(Q + A.T@PkOld@(A + B@L))

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

L = np.zeros((2,4)) #initialize policy
mu = np.zeros((2)) #initialize mu 
sigma = 0.1*np.eye(2) #initialize sigma using the identity matrix
alpha = 1e-13
G = 0 #initialize the reward counter 
timeCount = 0

for n in range(N):
    costs = []

    G = 0 #maybe reset G for every episode 
    
    x = dynfun.reset()
    for t in range(T):

        u = np.random.multivariate_normal(L@x, sigma)

        # get reward
        c = costfun.evaluate(x,u)

        G += c #update G. Maybe move outside loop
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)


        x = xp.copy()
        mu = 0.5*(mu + u)

    # TODO update policy
    gradLog = ((LA.inv(sigma) + LA.inv(sigma).T)@(L@x - mu)).reshape(2,1)@(x.reshape(4,1)).T #check dimensions of x.T

    L += alpha*G*gradLog
    #mu = 0.5*(mu + L@x) #update mu


    total_costs.append(sum(costs))

    if n == 0:
        LnormVal = LA.norm(L_star - L)
        normTime = np.array([timeCount])
        costHold = np.array([sum(costs)])
        timeCount += 1
    else:
        LnormVal = np.vstack((LnormVal, (LA.norm(L_star - L))))
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