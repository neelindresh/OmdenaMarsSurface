import cv2
import os

def drawObjects(real,filtered,imageName=None,lowerThes=120,upperThes=200,drawPoly=False,drawConvex=False):
	realImage=cv2.imread(real)
	#cv2.imshow('real',realImage)
	#cv2.waitKey()
	im=cv2.imread(filtered)
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thes=cv2.threshold(gray,lowerThes,upperThes,cv2.THRESH_BINARY_INV)
	_,countours,heir=cv2.findContours(thes,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for c in countours:
		x,y,w,h=cv2.boundingRect(c)
		if w*h >50:
			cv2.rectangle(realImage,(x,y),(x+w,y+h),(0,0,255),2)
	if drawPoly:
		for c in countours:
			accu=0.03*cv2.arcLength(c,True)
			approx=cv2.approxPolyDP(c,accu,True)
			cv2.drawContours(realImage,[approx],0,(0,255,0),2)
		#cv2.imshow('app',im)
	#cv2.waitKey()
	#Convex Hull
	if drawConvex:
		n=len(countours)-1
		countours=sorted(countours,key=cv2.contourArea,reverse=False)[:n]
		for c in countours:
			approx=cv2.convexHull(c)
			cv2.drawContours(realImage,[approx],0,(0,255,255),2)
		if imageName==None:
			imageName=str('detected'+real)
		print(imageName)
	cv2.imwrite(imageName,realImage)

for i in os.listdir('Crop3parts'):
	if i.endswith('.jpg'):
		if i.startswith('real'):
			realImage=i
			filterImage=i.split('.')[0]
			filterImage=filterImage.split('_')[1]+ '.jpg'
			print(realImage,filterImage)
			drawObjects('Crop3parts/'+realImage,'Crop3parts/'+filterImage,imageName='Crop3parts/rectOnly'+filterImage)
			drawObjects('Crop3parts/'+realImage,'Crop3parts/'+filterImage,imageName='Crop3parts/withconvex'+filterImage,drawConvex=True)
			drawObjects('Crop3parts/'+realImage,'Crop3parts/'+filterImage,imageName='Crop3parts/approx'+filterImage,drawPoly=True)
			drawObjects('Crop3parts/'+realImage,'Crop3parts/'+filterImage,imageName='Crop3parts/combined'+filterImage,drawPoly=True,drawConvex=True)