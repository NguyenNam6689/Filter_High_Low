import  numpy as np
import matplotlib.pyplot as plt
import  cv2
import math

#TH02_Translate
def image_translate (image,shiftX, shiftY):
    M = np.float32([[1,0,shiftX],[0,1,shiftY]])
    return cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))

# TH02_Rotation
def image_rotation(image,degree):
    theta=-degree*math.pi/180
    (h,w)=image.shape[:2]
    M=np.float32([[math.cos(theta), -math.sin(theta), 0],[math.sin(theta), math.cos(theta), 0]])
    return cv2.warpAffine(image, M,(w,h))

def remove_black_pixels(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    image[mask == 0] = [255, 255, 255]  # Set black pixels to white
    return image

image = cv2.imread('.venv/img/trex.png')
fig=plt.figure()
ax1,ax2=fig.subplots(1,2)
ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
ax1.set_title('original')

shifted=image_translate(image,10,20)
rotated=image_rotation(image,10)
shifted_rm = remove_black_pixels(shifted)
rotated_rm = remove_black_pixels(rotated)

ax2.imshow(rotated_rm)
ax2.set_title('Translate image')
plt.show()

