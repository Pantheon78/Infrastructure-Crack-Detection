import cv2 
import os
import numpy as np

img = cv2.imread('Crack Detection System/Images/Negative/00103.jpg')

img = cv2.resize(img,(0,0) ,fx = 0.8, fy = 0.8 )

#to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#blur image
blur = cv2.GaussianBlur(gray,(5,5), 0)

#edge detection 
edges = cv2.Canny(blur, 50 ,150)


#Morphological operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))#modifies pixels
dilation = cv2.dilate(edges,kernel,iterations=1)#makes white regions bigger 
closed = cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)#connects cracks 
erosion = cv2.erode(edges, kernel, iterations=1)

#finding contours
contours,_ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#filter and draw cracks
for cnt in contours:
    area = cv2.contourArea(cnt)#calculates how big  the shape is 

    #ignore small noise 
    if area < 150:
        continue
    x,y,w,h = cv2.boundingRect(cnt)

    #measures compactness
    perimeter = cv2.arcLength(cnt,True)
    if perimeter == 0:
        continue
    compactness = area / (perimeter * perimeter)

    if compactness < 0.02:
        cv2.drawContours(img,[cnt], -1,(0,150,0),2)
 
    #avoid division error
    if h == 0:
        continue  
    ratio = w / h

    #cracks are long and thin
    if ratio > 2 or ratio < 0.5:
        cv2.drawContours(img,[cnt], -1, (0,150,0),2)#draws green outline on img

cv2.imshow("Crack Detection",img)
cv2.imshow('Edges', edges)
cv2.imshow('dilation',dilation)


cv2.waitKey(0)
cv2.destroyAllWindows()

