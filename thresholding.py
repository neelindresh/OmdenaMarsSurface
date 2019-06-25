import cv2
import os
def featureExtraction(real,filtered,imageName=None,lowerThesBinary=120,upperThesBinary=200,lowerThesToZero=180,upperThesToZero=250,maxArea=200):
	realImage=cv2.imread(real)
	im=cv2.imread(filtered)
	
	cp_im=im.copy()
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

	ret,thes=cv2.threshold(gray,lowerThesBinary,upperThesBinary,cv2.THRESH_BINARY_INV)
	ret,thesNew=cv2.threshold(gray,lowerThesToZero,upperThesToZero,cv2.THRESH_TOZERO_INV)
	ret,thesNew=cv2.threshold(thesNew,lowerThesBinary,upperThesBinary,cv2.THRESH_BINARY_INV)
	

	#cv2.imshow('gray2',thesNew)
	bit_and=cv2.bitwise_or(thes,thesNew)

	_,countours,heir=cv2.findContours(bit_and,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for c in countours:
		x,y,w,h=cv2.boundingRect(c)
		if w*h >200:
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
		else:
			cv2.rectangle(bit_and,(x,y),(x+w,y+h),cv2.FILLED,cv2.FILLED)
	if imageName==None:
		imageName=str('detected'+real)
		print(imageName)
	cv2.imwrite(imageName,bit_and)
def featureExtractionv2(filtered,imageName=None,lowerThesBinary=120,upperThesBinary=200,lowerThesToZero=180,upperThesToZero=250,maxArea=200):
	#realImage=cv2.imread(real)
	im=cv2.imread(filtered)
	
	cp_im=im.copy()
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

	ret,thes=cv2.threshold(gray,lowerThesBinary,upperThesBinary,cv2.THRESH_BINARY_INV)
	ret,thesNew=cv2.threshold(gray,lowerThesToZero,upperThesToZero,cv2.THRESH_TOZERO_INV)
	ret,thesNew=cv2.threshold(thesNew,lowerThesBinary,upperThesBinary,cv2.THRESH_BINARY_INV)
	

	#cv2.imshow('gray2',thesNew)
	bit_and=cv2.bitwise_or(thes,thesNew)

	_,countours,heir=cv2.findContours(bit_and,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for c in countours:
		x,y,w,h=cv2.boundingRect(c)
		if w*h >200:
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
		else:
			cv2.rectangle(bit_and,(x,y),(x+w,y+h),cv2.FILLED,cv2.FILLED)
	if imageName==None:
		imageName=str('detected'+real)
		print(imageName)
	cv2.imwrite(imageName,bit_and)
'''
for i in os.listdir('Crop3parts'):
	if i.endswith('.jpg'):
		if i.startswith('real'):
			realImage=i
			filterImage=i.split('.')[0]
			filterImage=filterImage.split('_')[1]+ '.jpg'
			print(realImage,filterImage)
			featureExtraction('Crop3parts/'+realImage,'Crop3parts/'+filterImage,imageName='Crop3parts/extractedImage'+filterImage)
'''			