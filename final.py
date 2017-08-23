###################################################
# Author: Chirath Abeysinghe
# Email: lvchirathdevinda@gmail.com
###################################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2
import pandas as pd
import time

with open("records.txt", "w") as myfile1:
    myfile1.write("")

def plotCurve(E):

    # Take the Matrix to an array
    X = np.array(E)

    # Select only the points which are non zero and take their index coordinations such as Y and X position
    Y = np.transpose((X>0).nonzero())

    # Take the second column which is for X coordinates and first column for Y coordinates
    Z = np.concatenate(([Y[:, 1]],[Y[:, 0]]), axis=0)
    
    # Take the mean values for each X coordinates
    T = np.array(pd.DataFrame(Z.T).groupby(0).mean())

    # Normalise them into 100 scale
    T *= 100 / T.max()

    # Sort each X values in Z after selecting unique values
    Xhat = np.sort(np.unique(Z[0]))

    # Take the mean value of all the Y values
    Yhat = np.array([T.reshape(len(T)).mean()] * len(T))

    # Now we have a Horizontal Line which have the same Y value for each X values
    plt.plot(Xhat, Yhat)

    plt.gca().invert_yaxis()
    
    #plt.show()
    
    # Now we can return one of the Y values because all we need is a one point that represents the whole frame
    return Yhat[0]

def writeFile(MeanY0, MeanY1):
    X = str(int(round(time.time() * 1000)))
    
    with open("records.txt", "a") as myfile:
        myfile.write(X + "," + str(MeanY0) + "\n")
        
    with open("records1.txt", "a") as myfile:
        myfile.write(X + "," + str(MeanY1) + "\n")

# Load the video into system
cap = cv2.VideoCapture('LongVideo.mp4', 0)

# Creating a Loop that iterates the video sequence
while(cap.isOpened()):
    
    # Retrieve frame by frame
    ret, frame = cap.read()
    
    # Checking if the video has reached to the end if so brak the loop otherwise continue
    if frame is None:
        break
    
    # Take the Height and Width properties of the frame
    height, width, channels = frame.shape
    # Focus the upper part of the Signal
    frame0 = frame[0:height//2, 0:width]
    # Focus the lower part of the signal
    frame1 = frame[height//2:height, 0:width]

    # Add the GaussianBlur filter for noise reduction for both signals
    blur0 = cv2.GaussianBlur(frame0,(5,5),0)
    blur1 = cv2.GaussianBlur(frame1,(5,5),0)
    # Applying Canny Edge detection Algorithm
    edges0 = cv2.Canny(blur0,100,200)
    edges1 = cv2.Canny(blur1,100,200)
    # find thresholds of the edges
    ret0,thresh0 = cv2.threshold(edges0,127,255,0) 
    im20, contours0, hierarchy0 = cv2.findContours(thresh0,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour
    E0 = cv2.drawContours(edges0, contours0, -1, (255,255,0), 3)
    
    ret1,thresh1 = cv2.threshold(edges1,127,255,0)
    im21, contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    E1 = cv2.drawContours(edges1, contours1, -1, (255,255,0), 3)

    # Passing the sparse matrix into plotCurve function
    YCordinate0 = plotCurve(E0)
    YCordinate1 = plotCurve(E1)
    
    writeFile(YCordinate0, YCordinate1)
    
    cv2.namedWindow('Original')
    cv2.moveWindow('Original',frame.shape[1],10)
    cv2.namedWindow('Signal 1')
    cv2.moveWindow('Signal 1',E0.shape[1],380)
    cv2.namedWindow('Signal 2')
    cv2.moveWindow('Signal 2',E1.shape[1],420+E0.shape[0])

    cv2.imshow('Original', frame)
    cv2.imshow('Signal 1',E0)
    cv2.imshow('Signal 2',E1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
