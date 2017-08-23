import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2
import pandas as pd

def plotCurve(E):
    X = np.array(E)
    Y = np.transpose((X>0).nonzero())
    Z = np.concatenate(([Y[:, 1]],[Y[:, 0]]), axis=0)
    T = np.array(pd.DataFrame(Z.T).groupby(0).mean())
    T *= 100 / T.max()
    Xhat = np.sort(np.unique(Z[0]))
    #Yhat = np.array([T.reshape(len(T)).mean()] * len(T))
    plt.plot(Xhat, T)
    plt.ylim([0,100])
    plt.autoscale(False)
    plt.gca().invert_yaxis()
    plt.show()


cap = cv2.VideoCapture('LongVideo.mp4', 0)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if frame is None:
        break
    
    height, width, channels = frame.shape
    frame0 = frame[0:height/2, 0:width]
    frame1 = frame[height/2:height, 0:width]

        
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #image = Image.fromarray(gray)
    #image = image.filter(ImageFilter.FIND_EDGES)
    #image1 = np.array(image)
    
    #img = cv2.imread('C:\\Users\\Test\\Pictures\\Images\\wave.png',0)
    #blur = cv2.bilateralFilter(frame,9,75,75)
    blur0 = cv2.GaussianBlur(frame0,(5,5),0)
    blur1 = cv2.GaussianBlur(frame1,(5,5),0)
    #blur = cv2.bilateralFilter(frame,9,75,75)
    edges0 = cv2.Canny(blur0,100,200)
    edges1 = cv2.Canny(blur1,100,200)
    
    ret0,thresh0 = cv2.threshold(edges0,127,255,0)
    im20, contours0, hierarchy0 = cv2.findContours(thresh0,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    E0 = cv2.drawContours(edges0, contours0, -1, (255,255,0), 3)
    
    ret1,thresh1 = cv2.threshold(edges1,127,255,0)
    im21, contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    E1 = cv2.drawContours(edges1, contours1, -1, (255,255,0), 3)

    #plt.imshow(edges,cmap = 'gray')
    #plt.show()

    plotCurve(E0)
    
    #cv2.imshow('Original', frame)
    #cv2.imshow('Signal 0',E0)
    #cv2.imshow('Signal 1',E1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
