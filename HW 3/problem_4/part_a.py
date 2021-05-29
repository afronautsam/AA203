from model import dynamics, cost
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA

dynfun = dynamics(stochastic=False)
#dynfun = dynamics(stochastic=True) # uncomment for stochastic dynamics

costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

# Riccati recursion
def Riccati(A,B,Q,R):
    PkOld = np.zeros((len(A), len(A)))

    L = -inv(R + B.T@PkOld@B)@B.T@PkOld@A
    Pk = Q + A.T@PkOld@(A + B@L)

    while LA.norm(Pk - PkOld) > 1e-8:
        PkOld = Pk
        L = -inv(R + B.T@PkOld@B)@B.T@PkOld@A
        Pk = Q + A.T@PkOld@(A + B@L)
        #maybe update L again after

    
    return L,Pk


A = dynfun.A
B = dynfun.B
Q = costfun.Q
R = costfun.R

L,P = Riccati(A,B,Q,R)

total_costs = []

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):
        
        # policy 
        u = (L @ x)
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
    
        # dynamics step
        x = dynfun.step(u)
        
    total_costs.append(sum(costs))
    
print(np.mean(total_costs))