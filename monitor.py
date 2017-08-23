###################################################
# Author: Chirath Abeysinghe
# Email: lvchirathdevinda@gmail.com
###################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import pandas as pd
from numpy import linalg as LA
from numpy.linalg import norm
import math

X, Y, X1, Y1, X2, Y2, EigenvectorArray = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

fig = plt.figure()
fig.suptitle("MONITOR")
axplot1, axplot2, axplot3 = fig.add_subplot(3,1,1), fig.add_subplot(3,1,2), fig.add_subplot(3,1,3)

counter = 0

def logicalOR(inputData):
    return np.logical_or(np.insert(inputData, 0, 0) , np.append(inputData, 0)).astype(int)

def calculateAngle(V, U):
    v1 = np.linalg.norm(V)
    u1 = np.linalg.norm(U)
    return np.arccos(np.clip(np.dot(V, U) /v1 /u1, -1.0, 1.0))

def loadData(name):
    print("Came Here")
    raw_data = open(name, 'rb')
    data = np.loadtxt(raw_data, delimiter=',')[-100:]
    x = data[:,0]
    y = data[:,1]
    return x,y

def generateFrequency(x_values):
    return (1e3/np.diff(x_values)).tolist()

def frequencyDetector(freq):
    return logicalOR(np.array([1 if u >= 35 or u <= 10 else 0 for u in freq]))

def findCorrEigenVector(matrix):
    # Finding Eigenvector and corresponding Eigen values
    w, v = LA.eig(matrix)

    # Maximum eigenvalue and corrsponding eigen vector 
    return v[np.argmax(w, axis=0)].real

def plotGraph(x, y, freqIndicator, id):
    global axplot1, axplot2, axplot3

    # Color Map
    cMap = np.array(['b','r'])

    eval("axplot"+id).clear()

    axplot1.title.set_text('Signal 1')
    axplot2.title.set_text('Signal 2')
    axplot3.title.set_text('Signal 1')

    # Time series data range of last 100 data points
    eval("axplot"+id).set_ylim([0,100])
    eval("axplot"+id).set_xlim(x[:][0], x[:][-1])
    eval("axplot"+id).set_autoscale_on(False)

    eval("axplot"+id).plot(x,y, c="green", linewidth=3)
    eval("axplot"+id).plot([x*freqIndicator, x*freqIndicator+((np.roll(x*freqIndicator,-1)*freqIndicator) - x*freqIndicator)*np.roll(freqIndicator,-1)*freqIndicator],[y*freqIndicator, y*freqIndicator + ((np.roll(y*freqIndicator,-1)*freqIndicator) - y*freqIndicator)*np.roll(freqIndicator,-1)*freqIndicator], '-', c="red", linewidth=3)

def animate(i):
    global counter
    global EigenvectorArray

    # Resetting counter back to 0 otherwise eigenvalue array will be overflow
    if(counter==100):
        counter = 0

    # Showing only last 100 data points
    X, Y = loadData("records.txt")
    X1, Y1 = loadData("records1.txt")
    X2, Y2 = loadData("records.txt")
    
    # Forming a new Matrix based on 3 signals
    Matrix = np.vstack((Y.T,Y1.T,Y2.T))
    Matrix = Matrix.T.dot(Matrix)

    # Correcponding Maximum Eigen value associated Eigenvector
    Eigenvector = findCorrEigenVector(Matrix)
    
    # Add Eigenvector to the array
    EigenvectorArray = np.append(EigenvectorArray, Eigenvector)
    
    # Comparing angle between present and previous Eignevectors
    Theta = 0 if counter== 0 else calculateAngle(EigenvectorArray[counter], EigenvectorArray[counter-1])
    
    # Comparing angle between First and present Eignevectors
    Theta1 = 0 if counter== 0 else calculateAngle(EigenvectorArray[0], EigenvectorArray[counter])

    # Printing each angles
    print(math.degrees(Theta),math.degrees(Theta1))

    # Min Max Indicators
    # MinMaxIndicators = np.array([0 if counter < 80 and counter > 65 else 1 for counter in Y])

    frequencies1 = generateFrequency(X)
    freqIndicators1 = frequencyDetector(frequencies1)

    frequencies2 = generateFrequency(X1)
    freqIndicators2 = frequencyDetector(frequencies2)

    frequencies3 = generateFrequency(X2)
    freqIndicators3 = frequencyDetector(frequencies3)

    plotGraph(X, Y, freqIndicators1, '1')
    plotGraph(X1, Y1, freqIndicators2, '2')
    plotGraph(X2, Y2, freqIndicators3, '3')

    #ax.scatter(X,Y, c=cMap[MinMaxIndicators])

    counter += 1
    
ani = animation.FuncAnimation(fig, animate, blit=False, interval=10, repeat=False)
axplot1.plot(X,Y)
axplot2.plot(X1,Y1)
axplot3.plot(X2,Y2)

plt.show()
