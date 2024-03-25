import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

#ham bien doi anh dao dau
def image_negative(img):
    return (255-img)

def log_transformation(img,c=1):
    return float(c)*cv.log(1.0+img)

#ham bien doi gamma
def gamma_transformation ( img , gamma , c=1) :
    return float(c)*pow(img,float(gamma))

def thresholding_binary(img,m):
    return img>m

img=cv.imread('./.venv/img/trex.png',0)

img1=image_negative(img)
img2=log_transformation(img,3)
img3=gamma_transformation(img,15)
img4=thresholding_binary(img,2)

fig=plt.figure()
ax1,ax2=fig.subplots(1,2)
#hien thi anh goc
ax1.imshow(img)
ax1.set_title('Original')
#hien thi anh bien doi
ax2.imshow(img4)
ax2.set_title('Transform')
plt.show()

