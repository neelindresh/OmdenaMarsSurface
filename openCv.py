import cv2
import os

realImage=cv2.imread('Crop3parts/real2.jpg')
im=cv2.imread('Crop3parts/2.jpg')
#cv2.imshow('actual',im)
#cv2.waitKey()
cp_im=im.copy()
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
canny=cv2.Canny(gray,50,50)


ret,thes=cv2.threshold(gray,120,200,cv2.THRESH_BINARY_INV)
#cv2.imshow('gray',thes)
#cv2.waitKey()
_,countours,heir=cv2.findContours(thes,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for c in countours:
	x,y,w,h=cv2.boundingRect(c)
	if w*h >50:
		cv2.rectangle(realImage,(x,y),(x+w,y+h),(0,0,255),2)
	#cv2.imshow('BI',im)
#cv2.waitKey()

for c in countours:
	accu=0.03*cv2.arcLength(c,True)
	approx=cv2.approxPolyDP(c,accu,True)
	#cv2.drawContours(im,[approx],0,(0,255,0),2)
	#cv2.imshow('app',im)
#cv2.waitKey()
#Convex Hull
n=len(countours)-1
countours=sorted(countours,key=cv2.contourArea,reverse=False)[:n]
for c in countours:
	approx=cv2.convexHull(c)
	cv2.drawContours(realImage,[approx],0,(0,255,255),2)
	cv2.imshow('app',im)
cv2.waitKey()
cv2.imwrite('0contour.jpg',realImage)