import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt 
from matplotlib.pyplot import ion
import seaborn as sns


class droneMDP:
    def __init__(self, disc, n):
        self.disc = disc #defining the discount factor
        self.stateGrid = np.mgrid[0:n, 0:n] #create a 2xn-1xn-1 meshgrid to represent the statespace 
        self.actionSpace = {"right" : [0, 1], "left": [0, -1], "down":[-1, 0], "up":[1, 0]} #defining the action space as a dict for convenience. Arrays can be added
        self.reward = np.zeros((n,n)) #review, should it be 3d or 2D? 
        self.transition = np.zeros((n,n))
        self.value = np.zeros((n,n)) #haha :(
        #self.prevVal = np.zeros((n,n))
        self.optPolicyName = [["foo" for i in range(n)] for j in range(n)]
        self.optPolicyName = np.array(self.optPolicyName) #forcing policy name to be a numpy array
        self.optPolicyVal = np.zeros((n,n))


    def inSpace(self, sVal):
        bigS = self.stateGrid
        sX = sVal[0] #state values of interest 
        sY = sVal[1]

        bigSx = bigS[0, :, :] #splitting the state mesh into x & y
        bigSy = bigS[1, : ,:]
        flagVal = False
        for x,y in np.ndindex(bigSx.shape):
            if bigSx[x, y] == sX and bigSy[x,y] == sY: #ensuring the provided state is in the state grid 
                flagVal = True
        return flagVal

    def initProbDist(self, sEye, sigma):
        bigS = self.stateGrid
        bigSx = bigS[0,:,:]
        bigSy = bigS[1,:,:]
        T = self.transition

        for s in np.ndindex(bigSx.shape): #iterating over every state 
            sArray = np.asarray(s)
            xMat = (LA.norm(np.subtract(sArray, sEye))) ** 2 #computing the square of the 2-norm
            T[s] = np.exp(-xMat/(2*np.square(sigma)))

        return T

    def lookahead(self,s):
        T, R, gamma, A, U = self.transition, self.reward, self.disc, self.actionSpace, self.value
        
        maxVal = 0 #initializing variable used to capture sum
        optPol = 0

        for key in A:
            if key == "up":
                sDes = tuple(np.add(s, A["up"]))
                sRand1 = tuple(np.add(s, A["down"]))
                sRand2 = tuple(np.add(s, A["left"]))
                sRand3 = tuple(np.add(s, A["right"]))
            elif key == "down":
                sDes = tuple(np.add(s, A["down"]))
                sRand1 = tuple(np.add(s,A["up"]))
                sRand2 = tuple(np.add(s, A["left"]))
                sRand3 = tuple(np.add(s, A["right"]))
            elif key == "left":
                sDes = tuple(np.add(s, A["left"]))
                sRand1 = tuple(np.add(s, A["up"]))
                sRand2 = tuple(np.add(s, A["down"]))
                sRand3 = tuple(np.add(s, A["right"]))
            elif key == "right":
                sDes = tuple(np.add(s, A["right"]))
                sRand1 = tuple(np.add(s, A["up"]))
                sRand2 = tuple(np.add(s, A["down"]))
                sRand3 = tuple(np.add(s, A["left"]))
            
            
            pDist = (1 - T[s])+T[s]/4 #probability of moving in the specified direction (regardless of what it is )
            randProb1 = T[s]/4 #probability of moving in a random direction 
            randProb2 = T[s]/4
            randProb3 = T[s]/4

            if self.inSpace(sDes) == False: #if the action is to move off the state grid, don't 
                sDes = s
            if self.inSpace(sRand1) == False:
                sRand1 = s
            if self.inSpace(sRand2) == False:
                sRand2 = s
            if self.inSpace(sRand3) == False:
                sRand3 = s
            
            sumVal = pDist*(R[sDes] + gamma*U[sDes]) + randProb1*(R[sRand1] + gamma*U[sRand1]) + randProb2*(R[sRand2] + gamma*U[sRand2]) + randProb3*(R[sRand3] + gamma*U[sRand3])

            if sumVal > maxVal:
                maxVal = sumVal #return the maxiumum value function 
        
        return maxVal
    
    def valueIteration(self): #too many loops. Basically doing a 50x400 loop 
        for i in range(300):
            redStateGrid = self.stateGrid[1,:,:]

            for s in np.ndindex(redStateGrid.shape):
                self.value[s] = self.lookahead(s)
        
        return self
    
    def computeOptPolicy(self):
        A = self.actionSpace
        for s in np.ndindex(self.value.shape):
            uMax = float('-inf')
            for key in self.actionSpace:
                if key == "up":
                    sNew = tuple(np.add(s, A["up"]))
                    keyNum = 0
                    if self.inSpace(sNew)== False:
                        sNew = s
                        #uNew = -1
                    uNew = self.value[sNew]
                elif key == "down":
                    sNew = tuple(np.add(s, A["down"]))
                    keyNum = 1
                    if self.inSpace(sNew)== False:
                        sNew = s
                        #uNew = -1
                    uNew = self.value[sNew]
                elif key == "left":
                    sNew = tuple(np.add(s, A["left"]))
                    keyNum = 2
                    if self.inSpace(sNew)== False:
                        sNew = s
                        #uNew = -1
                    uNew = self.value[sNew]
                elif key == "right":
                    sNew = tuple(np.add(s, A["right"]))
                    keyNum = 3
                    if self.inSpace(sNew)== False:
                        sNew = s
                        #uNew = -1
                    uNew = self.value[sNew]

                if uNew > uMax:
                    uMax = uNew
                    self.optPolicyVal[s] = keyNum

        return self

    def simulate(self, sInit):
        A, T, optPolicy = self.actionSpace, self.transition, self.optPolicyVal 
        sNew = sInit #the initial/starting space 
        actDict = {3 : 0, 2: 0, 1: 0, 0: 0} #dictionary of actions/initial probabilities 
        actList = np.array([3, 2, 1, 0]) #list of actions
        stateStore = np.zeros([100,2]) #store the state followed along the path
        #sNew = np.zeros([2,])

        for t in range(100):
            sCurr = sNew
            optAct = optPolicy[sCurr] #determine the optimal action at this time step
 
            for key in actDict: #iterate over all the actions in the actionspace
                if key == optAct: #if the action is the optimal action, update the probability
                    actDict[key] = (1 - T[sCurr]) + T[sCurr]/4 #probability of taking the action given the state
                else: #else, use the random probability
                    actDict[key] = T[sCurr]/4 #probability of taking a random action 

            probDist = list(actDict.values()) #extract a list of probabilities corespondding to each action 
            actNew = np.random.choice(actList, p = probDist) #randomly select an action 
            
            if actNew == 3: #update the state based on the action 
                sNew = tuple(np.add(sCurr, A["right"]))
            elif actNew == 2:
                sNew = tuple(np.add(sCurr, A["left"]))
            elif actNew == 1:
                sNew = tuple(np.add(sCurr, A["down"]))
            elif actNew == 0:
                sNew = tuple(np.add(sCurr, A["up"]))
            
            if self.inSpace(sNew) == False:
                sNew = sCurr
            stateStore[t] = sNew
        return stateStore





def main():
    n = 20
    sigma = 10
    gamma = 0.95
    xEye = np.array([15, 15])
    xGoal = np.array([9, 19])
    xInit = (19,0)

    drone = droneMDP(gamma, n)
    drone.transition = drone.initProbDist(xEye, sigma) #initialize the transition matrix
    drone.reward[xGoal[0], xGoal[1]] = 1
    drone.valueIteration()
    drone.computeOptPolicy()
    traj = drone.simulate((xInit))

    plt.figure()
    ax = sns.heatmap(drone.value, linewidth=0.5, cmap="YlGnBu", linecolor="Black")
    ax.invert_yaxis() #get the plot to look right
    ax.set_title("Optimized Value Function ")
    plt.figure()
    ax2 = sns.heatmap(drone.optPolicyVal, linewidth=0.5, cmap="YlGnBu", linecolor="Black")
    ax2 = sns.scatterplot(traj[:,1], traj[:,0])
    ax2.invert_yaxis() #get the plot to look right
    ax2.set_title("Optimal Policy at Each Timestep with Simulated Trajectory")

    plt.show()
    #traj = drone.simulate((xInit))

    ray = 2 + 1



if __name__ == "__main__":
    main()    