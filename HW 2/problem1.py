from problem1_q_learning_env import *
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import time 
sim = simulator()

T = 5*365 # simulation duration
gamma = 0.95 # discount factor
eps = 0 #random exploration probability 
alpha = 0.95 #decay rate 


# get historical data
data = generate_historical_data(sim)
# historical dataset: 
# shape is 3*365 x 4
# k'th row contains (x_k, u_k, r_k, x_{k+1})

class QLearning:

    def __init__(self, gamma): #initialize the q learning class
        self.stateSpace = np.arange(6) #assumes state space is a 1D array
        self.actSpace = np.array(([0, 2, 4])) #assumes action space is a 1D array
        self.actDict = {0:0, 2:1, 4:2} #dict mapping action to indices
        self.Q = np.zeros((len(self.stateSpace), len(self.actSpace))) #initialize the Q matrix
        self.discount = gamma 
        self.alpha = 0.05 #modify learning rate as needed

    def update(self, s, a, r, sNew):
        gamma, Q, alpha = self.discount, self.Q, self.alpha 
        s = int(s)
        a = int(a)
        sNew = int(sNew)
        aInd = self.actDict[a] #getting the appropriate index for the specified action
        Q[s, aInd] += alpha*(r + gamma*np.amax(Q[sNew,:]) - Q[s, aInd]) #update the Q function
        self.Q = Q

        return self #returns the Q structure 

class valueIteration:
    def __init__(self, gamma):
        self.discount = gamma
        self.stateSpace = np.arange(6)
        self.actSpace = np.array(([0, 2, 4])) #assumes action space is a 1D array
        self.actDict = {0:0, 2:1, 4:2} #dict mapping action to indices
        self.actDictR = {0:0, 1:2, 2:4} #opposite mapping for value iteration 
        self.Q = np.zeros((len(self.stateSpace), len(self.actSpace))) #initialize the Q matrix
        self.value = np.zeros((len(self.stateSpace)))

    def lookahead(self, sA):
        s = sA[0]
        aVal = sA[1]

        a = self.actDictR[aVal]

        Q, bigS, gamma, bigA, V = self.Q, self.stateSpace, self.discount, self.actSpace, self.value

        pMat = [0.1, 0.3, 0.3, 0.2, 0.1]

        newVal = ((pMat[0]*(sim.get_reward(s, a, 0) + gamma*V[sim.transition(s, a, 0)])) + 
                    (pMat[1]*(sim.get_reward(s, a, 1) + gamma*V[sim.transition(s, a, 1)])) + 
                    (pMat[2]*(sim.get_reward(s, a, 2) + gamma*V[sim.transition(s, a, 2)])) + 
                    (pMat[3]*(sim.get_reward(s, a, 3) + gamma*V[sim.transition(s, a, 3)])) + 
                    (pMat[4]*(sim.get_reward(s, a, 4) + gamma*V[sim.transition(s, a, 4)])))


        
        return newVal #returns the value of the state and action pair

    def iterate(self):
        for i in range(150):
            for sA in np.ndindex(self.Q.shape):
                v = self.lookahead(sA)
                self.Q[sA] = v
                if v > self.value[sA[0]]: #update the value matrix 
                    self.value[sA[0]] = v

 

def policy(state,QStruct):
    maxInd = np.argmax(QStruct.Q[state, :])
    greedyAct = QStruct.actSpace[maxInd]
    return greedyAct      #return a greedy action 

    
#def train(QStruct)

def main():
  
    QVal = QLearning(gamma) #create an instance of the q learning class

    fig0 = plt.figure() #declaring figures for the qlearning training portion 
    ax0 = fig0.add_subplot(1,1,1)
    ax0.set_xlabel("Number of Iterations")
    ax0.set_ylabel("Magnitude of Q Value")


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel("Number of Iterations")
    ax1.set_ylabel("Magnitude of Q Value")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_xlabel("Number of Iterations")
    ax2.set_ylabel("Magnitude of Q Value")

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.set_xlabel("Number of Iterations")
    ax3.set_ylabel("Magnitude of Q Value")

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1,1,1)
    ax4.set_xlabel("Number of Iterations")
    ax4.set_ylabel("Magnitude of Q Value")

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(1,1,1)
    ax5.set_xlabel("Number of Iterations")
    ax5.set_ylabel("Magnitude of Q Value")

    xArr = []
    numIters = 100
    yArr = np.zeros((6,3,numIters))

    for i in range(numIters): #Training the QLearning Model
        for j in range(len(data)):
            newRow = data[j, :]
            s = newRow[0]
            a = newRow[1]
            r = newRow[2]
            sNew = newRow[3]

            QVal.update(s, a, r, sNew)

        
        QMat = QVal.Q
        xArr.append(int(i))
        yArr[:,:,i] = QMat

    for i in range(6):
        for j in range(3):
            if i == 0:
                ax0.plot(xArr, yArr[i,j,:])
            elif i == 1:
                ax1.plot(xArr, yArr[i,j,:])
            elif i == 2:
                ax2.plot(xArr, yArr[i,j,:])
            elif i == 3:
                ax3.plot(xArr, yArr[i,j,:])
            elif i == 4:
                ax4.plot(xArr, yArr[i,j,:])
            elif i == 5:
                ax5.plot(xArr, yArr[i,j,:])
        
        if i == 0:
            ax0.legend(['0', '2', '4'])
        elif i == 1:
            ax1.legend(['0', '2', '4'])
        elif i == 2:
            ax2.legend(['0', '2', '4'])
        elif i == 3:
            ax3.legend(['0', '2', '4'])
        elif i == 4:
            ax4.legend(['0', '2', '4'])
        elif i == 5:
            ax5.legend(['0', '2', '4'])
            
    valQ = valueIteration(gamma)
    valQ.iterate()

    print("Q-Learning:", QVal.Q)
    print("Value Iteration:", valQ.Q)

    # Forward simulating the system 
    rMat = np.zeros(T)
    aMat = np.zeros(T)
    s = sim.reset()
    for t in range(T):
        a = policy(s,valQ) #modification to pass the class instead of just the matrix
        sp,r = sim.step(a)
        s = sp
        aMat[t] = a
        if t == 0:
            rMat[t] = r
        else:
            rMat[t] = r + rMat[t-1]
    t = np.arange(T)
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(1,1,1)
    ax6.set_xlabel("Number of Days")
    ax6.set_ylabel("Aggregate Reward")
    ax6.plot(t, rMat)

    kroo = 1+2
    plt.show()



# TODO: write value iteration to compute true Q values
# use functions:
# - sim.transition (dynamics)
# - sim.get_reward 
# plus sim.demand_probs for the probabilities associated with each demand value


if __name__ == "__main__":
    main()    

