import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt 
import seaborn as sns

class droneMDP:
    def __init__(self, disc, n):
        self.disc = disc #defining the discount factor
        self.stateGrid = np.mgrid[0:n, 0:n] #create a 2xn-1xn-1 meshgrid to represent the statespace 
        # steps = np.linspace(0, n, n, endpoint=False)
        # intStateGrid = np.zeros((2,n,n))
        # stateX,stateY = np.meshgrid(steps, steps, sparse=False, indexing='xy')
        # intStateGrid[0,:,:] = stateX
        # intStateGrid[1,:,:] = stateY
        #self.stateGrid = intStateGrid
        self.actionSpace = {"up" : [0, 1], "down": [0, -1], "left":[-1, 0], "right":[1, 0]} #defining the action space as a dict for convenience. Arrays can be added
        self.reward = np.zeros((n,n)) #review, should it be 3d or 2D? 
        self.transition = np.zeros((n,n))
        self.value = np.zeros((n,n)) #haha :(
        self.optPolicyName = [["foo" for i in range(n)] for j in range(n)]
        self.optPolicyName = np.array(self.optPolicyName)
        self.optPolicyVal = np.zeros((n,n))

    def inSpace(self, sVal):
        bigS = self.stateGrid
        sX = sVal[0] #state values of interest 
        sY = sVal[1]

        bigSx = bigS[0, :, :] #splitting the state mesh into x & y
        bigSy = bigS[1, : ,:]
        flagVal = False
        for x,y in np.ndindex(bigSx.shape):
            if bigSx[x, y] == sX and bigSy[x,y] == sY: #ensuring the 
                flagVal = True
        return flagVal


    def transFunc(self, sEye, sigma):
        bigS = self.stateGrid
        bigSx = bigS[0,:,:]
        bigSy = bigS[1,:,:]
        T = self.transition

        for s in np.ndindex(bigSx.shape):
            sArray = np.asarray(s)
            xMat = (LA.norm(np.subtract(sArray, sEye))) ** 2
            T[s] = np.exp(-xMat/(2*np.square(sigma)))

        return T


    def lookahead(self, s, a):
        T, R, discount, A, U = self.transition, self.reward, self.disc, self.actionSpace, self.value
        sumVal = 0
        sX = s[0]
        sY = s[1]

        actNum = a[0]
        actKey = a[1] 

        for key in A:
            if key == "up":
                sNew = tuple(np.add(s, A["up"]))
            elif key == "down":
                sNew = tuple(np.add(s, A["down"]))
            elif key == "left":
                sNew = tuple(np.add(s, A["left"]))
            elif key == "right":
                sNew = tuple(np.add(s, A["right"]))
            
            if key == actKey:
                Tval = 1 - T[s]
            else:
                Tval = (T[s])/3
            
            if self.inSpace(sNew) == False:
                sNew = s
            
            sumVal += Tval*U[sNew] #assume a is a number and s is a tuple
        
        totalSum = R[actNum, sX, sY] + discount*sumVal
        return totalSum


    def backup(self, s):
        uMax = float('-inf')
        U = self.value
        A = self.actionSpace
        for key in A:
            if key == "up":
                actVec = [0, key]
            elif key == "down":
                actVec = [1, key]
            elif key == "left":
                actVec = [2, key]
            else:
                actVec = [3, key]
            uVal = self.lookahead(s, actVec)
            if uVal > uMax:
                uMax = uVal
                actMax = actVec
        return uMax, actMax[0], actMax[1]

    
    def valueIteration(self):
        for i in range(300): #replace with  while loop or for loop with ~250 iterations 
            U = self.value
            piN = self.optPolicyName
            pi = self.optPolicyVal
            stateSpaceAll = self.stateGrid
            stateSpace = stateSpaceAll[1,:,:]
            for s in np.ndindex(stateSpace.shape):
                U[s], pi[s], piN[s] = self.backup(s)
                self.value[s] = U[s]
                self.optPolicyName[s] = piN[s]
                self.optPolicyVal[s] = pi[s]

        return self
    

def computePolicy(drone, n):
    A, U= drone.actionSpace, drone.value
    policyName = [["foo" for i in range(n)] for j in range(n)]
    policyVal = np.zeros([n,n])

    #policy = []
    policyName = np.array(policyName)

    for s in np.ndindex(U.shape):
        uMax = float('-inf')
        for key in A:
            if key == "up":
                sNew = tuple(np.add(s, A["up"]))
                keyNum = 0
                if drone.inSpace(sNew)== False:
                    sNew = s
                    uNew = -1
                else:
                    uNew = U[sNew]
            elif key == "down":
                sNew = tuple(np.add(s, A["down"]))
                keyNum = 1
                if drone.inSpace(sNew)== False:
                    sNew = s
                    uNew = -1
                else:
                    uNew = U[sNew]
            elif key == "left":
                sNew = tuple(np.add(s, A["left"]))
                keyNum = 2
                if drone.inSpace(sNew)== False:
                    sNew = s
                    uNew = -1
                else:
                    uNew = U[sNew]
            elif key == "right":
                sNew = tuple(np.add(s, A["right"]))
                keyNum = 3
                if drone.inSpace(sNew)== False:
                    sNew = s
                    uNew = -1
                else:
                    uNew = U[sNew]
            
            if uNew > uMax:
                uMax = uNew
                policyName[s] = key
                policyVal[s] = keyNum
    
    return policyName, policyVal


def main():
    n = 20
    sigma = 10
    gamma = 0.95
    xEye = np.array([15, 15])
    xGoal = np.array([9, 19])

    drone = droneMDP(gamma, n)
    drone.transition = drone.transFunc(xEye, sigma)
    drone.reward[:,xGoal[0], xGoal[1]] = 1
    drone.valueIteration()

    #drone.value = drone.value.T #transpose 
    #drone.value = np.eye(n)[::-1]@drone.value #change to cartesian coords 
    #optPolicyName, optPolicyVal = computePolicy(drone, n) #fix policy. Plot directions 
    plt.figure()
    ax = sns.heatmap(drone.value, linewidth=0.5, cmap="YlGnBu", linecolor="Black")

    #optPolicyName, optPolicyVal = optPolicyName.T, optPolicyVal.T
    #optPolicyVal = np.eye(n)[::-1]@optPolicyVal
    plt.figure()
    ax2 = sns.heatmap(drone.optPolicyVal, linewidth=0.5, cmap="YlGnBu", linecolor="Black")
    ax.invert_yaxis() #get the plot to look right
    ax2.invert_yaxis()

    plt.show()

    

    ray = 2 + 1
if __name__ == "__main__":
    main()    