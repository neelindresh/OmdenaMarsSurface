import numpy as np
import cv2
import math
from scipy import ndimage
import matplotlib.pyplot as plt
from lxml import etree 
import xml.etree.cElementTree as xml
import os
from PIL import Image
"""Tools for satellite imagery pre-processing"""

def sharp(img,level=3): #level[1:5]
    
    def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
    
    if level == 1: #low
        sharpened = unsharp_mask(img)
        
    elif level == 2: #med_low
        kernel_sharp = np.array([[0, -1, 0], 
                                 [-1, 5, -1], 
                                 [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel_sharp)
        #sharpened = cv2.bilateralFilter(sbgimg, 3, 75 ,75)

    elif level == 3: #med. Best result on average
        kernel_sharp = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel_sharp)

    elif level == 4: #med_high
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)

    elif level == 5: #high
        kernel_sharp = np.array([[-2, -2, -2], 
                                 [-2, 17, -2], 
                                 [-2, -2, -2]])
        sharpened = cv2.filter2D(img, -1, kernel_sharp)
    
    else:
        sharpened = img
        print("image didn't change...")
    
    return sharpened
def thresholdingTOZERO(image,lower=100,upper=200):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thes=cv2.threshold(gray,lower,upper,cv2.THRESH_TOZERO)
    return thes
def thresholdingBINARY(image,lower=100,upper=200):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thes=cv2.threshold(gray,lower,upper,cv2.THRESH_BINARY_INV)
    return thes
def rectCoun(image,mode='bi',lower=None,upper=None,maxdim=4096,maxwidth=1000,maxheight=1000):
    '''
    upper=Upper limit of threshold
    lower=Lower limit of threshold
    maxdim=Maximum dimention example (64*64)==4096
     '''
    listoflocations=[]
    if lower==None: lower=100
    if upper==None: upper=200
    if mode=='bi':
        
        thes=thresholdingBINARY(image,lower,upper)
    elif mode=='zero':
        thes=thresholdingTOZERO(image,lower,upper)
    _,countours,heir=cv2.findContours(thes,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for c in countours:
        temp_={}
        x,y,w,h=cv2.boundingRect(c)
        if w*h >maxdim and w<maxwidth and h<maxheight:
            if w>h:
                w=h
            else:
                h=w
            temp_['xmin']=x
            temp_['xmax']=x+w
            temp_['ymin']=y
            temp_['ymax']=y+h
            listoflocations.append(temp_)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    return image,listoflocations
def check_and_create(path):
    if not os.path.exists(path):
        print('Folder Doesnot Exist! creating new folder: '+path)
        os.mkdir(path)
def write_to_xml(folder,image,objects,savedir):
    path=folder+image
    check_and_create(savedir)
    img=cv2.imread(path)
    height,width,depth=img.shape
    annotation=xml.Element('annotation')
    xml.SubElement(annotation,'folder').text=folder
    xml.SubElement(annotation,'filename').text=path
    xml.SubElement(annotation,'segmented').text='0'
    size=xml.SubElement(annotation,'size')
    xml.SubElement(size,'width').text=str(width)
    xml.SubElement(size,'height').text=str(height)
    xml.SubElement(size,'depth').text=str(depth)
    for coor in objects:
        ob=xml.SubElement(annotation,'object')
        xml.SubElement(ob,'name').text='techno-signature'
        xml.SubElement(ob,'pose').text='Unspecified'
        xml.SubElement(ob,'truncated').text='0'
        xml.SubElement(ob,'difficult').text='0'
        bbox=xml.SubElement(ob,'bndbox')
        xml.SubElement(bbox,'xmin').text=str(coor['xmin'])
        xml.SubElement(bbox,'xmax').text=str(coor['xmax'])
        xml.SubElement(bbox,'ymin').text=str(coor['ymin'])
        xml.SubElement(bbox,'ymax').text=str(coor['ymax'])
    xml_string=xml.tostring(annotation)
    root=etree.fromstring(xml_string)
    xml_string=etree.tostring(root,pretty_print=True)
    save_xml(xml_string,savedir,image)
    return xml_string
def save_xml(xmlstring,savedir,filename):
    filename=filename.split('.')[0]
    with open(savedir+filename+'.xml','wb') as xml:
        xml.write(xmlstring)
def save(savedir,filename,image):
    check_and_create(savedir)
    cv2.imwrite(savedir+filename,image)
def featureExtractionv2(filtered,imageName=None,lowerThesBinary=120,upperThesBinary=200,lowerThesToZero=180,upperThesToZero=250,maxArea=200,write=True):
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
    if write : 
        if imageName==None:
            imageName=str('detected'+real)
            print(imageName)
        cv2.imwrite(imageName,bit_and)
    return bit_and
def featureDetectionAndRectangle(im,maxfeatures=5,maxdim=4096,maxwidth=1000,maxheight=1000):
    #gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thes=cv2.threshold(im,120,200,cv2.THRESH_BINARY_INV)
    _,countours,heir=cv2.findContours(thes,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    size={}
    counter=0
    coor={}
    for c in countours:
        x,y,w,h=cv2.boundingRect(c)
        if w*h >maxdim and w<maxwidth and h<maxheight:
            if w>h:
                w=h
            else:
                h=w

            size['img_'+str(counter)]=w*h
            coor['img_'+str(counter)]={'xmin':x,'xmax':x+w,'ymin':y,'ymax':y+h}
            counter+=1
        
        #cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
    sorted_x = sorted(size.items(), key=lambda kv: kv[1],reverse=True)
    listoflocations=[]
    for i in sorted_x[:maxfeatures]:
        coor[i[0]]
        listoflocations.append(coor[i[0]])
    return listoflocations
def crop(filename,savedir,width,height,resize=None):
    check_and_create(savedir)
    im=Image.open(filename)
    if resize !=None:
        im=im.resize(resize)
    imwidth,imheight=im.size
    name=filename.split('/')[-1]
    name=name.split('.')[0]
    count=0
    for i in range(0,imwidth,width):
        for j in range(0,imheight,height):
            #print(j)
            box=(i,j,i+width,j+height)
            img=im.crop(box)
            img.save(savedir+'/'+name+'_'+str(count)+'.jpg')
            count+=1