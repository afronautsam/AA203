import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 


def yDot(y, t, u):
    dY = y + 3*u
    
    return dY

def ymDot(ym, t, r):
    dY = -4*ym + 4*r
    
    return dY



def uNew(kR, r, kY, y):
    uVal = kR*r + kY*y

    return uVal

def rNew(t, tag):
    if tag == 1:
        rNewVal =  4
    elif tag == 2:
        rNewVal = 4*np.sin(3*t)

    return rNewVal

def error(e, y):
    alphaM = 4
    betaM = 4

    dE = - e * alphaM + y
    
def simulate(ymDot, yDot, rNew, uNew):
    t = np.linspace(0,10,100)
    y = np.zeros((len(t),1))
    ym = np.zeros((len(t),1))
    r = np.zeros(len(t))
    u = np.zeros(len(t))
    e = np.zeros(len(t))
    kR = np.zeros(len(t))
    kY = np.zeros(len(t))
    y0 = 0
    ym0 = 0
    gamma = 2
    beta = 3



    for i in range(len(t)):
        r[i] = rNew(t[i], tag=2)
        ym[i:i+2] = odeint(ymDot, ym[i], t[i:i+2], args=(r[i], ))

        e[i] = y[i] - ym[i]
        kR[i] = kR[i-1] -np.sign(beta)*gamma*e[i]*r[i]*0.1
        kY[i] = kY[i-1] -np.sign(beta)*gamma*e[i]*y[i]*0.1
        u[i] = uNew(kR[i], r[i], kY[i], y[i])

        y[i:i+2] = odeint(yDot, y[i], t[i:i+2], args=(u[i],) )


    return t,y, ym, kR, kY


def main():
    alpha = -1
    beta = 3

    alphaM = 4
    betaM = 4
    gamma = 2

    t, y, ym, kR, ky = simulate(ymDot, yDot, rNew, uNew)

    kRstar = np.ones(100) * (betaM/beta)
    kYstar = np.ones(100) * (alpha - alphaM)/beta

    fig1,ax1 = plt.subplots()
    ax1.plot(t, y,label='Model')
    ax1.plot(t, ym, label='reference')
    ax1.set_xlabel('Timesteps (in seconds)')
    ax1.set_ylabel('State Variable Magnitude')
    ax1.legend()
    ax1.set_title('MRAC with Periodic Reference Dynamics')

    fig2,ax2 = plt.subplots()
    ax2.plot(t, kR,label='kR')
    ax2.plot(t, ky, label='kY')
    ax2.plot(t, kRstar,label='kR*')
    ax2.plot(t, kYstar,label='ky*')
    ax2.set_xlabel('Timesteps (in seconds)')
    ax2.set_ylabel('Controller Gain Magnitude')
    ax2.legend()
    ax2.set_title('MRAC Gains with Periodic Reference Dynamics')


    plt.show()
    ray = 2 + 1



if __name__ == "__main__":
    main()    

