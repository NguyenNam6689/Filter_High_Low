import numpy as np
import matplotlib.pyplot as plt
import cv2

class lowPassFilter():
    '''Lớp xử lý mịn ảnh'''
    def __init__(self,img) -> None:
        self.img = img
    def convolution2d(self,kernel):
        m,n = self.img.shape
        img_new = np.zeros([m,n],dtype="uint8")
        for i in range(1,(m-1)):
            for j in range(1,(n-1)):
                temp = self.img[i-1,j-1] * kernel[0,0]\
                + self.img[i-1,j]        * kernel[0,1]\
                + self.img[i-1,j+1]      * kernel[0,2]\
                + self.img[i, j-1]       * kernel[1,0]\
                + self.img[i, j]         * kernel[1,1]\
                + self.img[i, j+1]       * kernel[1,2]\
                + self.img[i+1,j-1]      * kernel[2,0]\
                + self.img[i+1,j]        * kernel[2,1]\
                + self.img[i+1,j+1]      * kernel[2,2]
                img_new[i,j] = temp
        img_new = img_new.astype(np.uint8)
        return img_new
    def meanFilter(self):
        MeanKernel33 = np.array([[1/9,1/9,1/9],
                                    [1/9,1/9,1/9],
                                    [1/9,1/9,1/9]],dtype="float")
        return self.convolution2d(MeanKernel33)
    def GaussFilter(self):
        GaussKernel33 = np.array(([0.3679/4.8976,0.6065/4.8976, 0.3679/4.8976],
                                  [0.6065/4.8976, 1.0/4.8976, 0.6065/4.8976],
                                  [0.3679/4.8976, 0.6065/4.8976, 0.3679/4.8976]),dtype="float")
        return self.convolution2d(GaussKernel33)

    def median_Filter(img):
        return cv2.medianBlur(img, 9)

    def box_Filter(img):
        return cv2.boxFilter(img, -1, (10, 10), normalize=True)

if __name__ == '__main__':
    img = cv2.imread('./.venv/img/test2.tif', cv2.IMREAD_GRAYSCALE)
    fig = plt.figure()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('original')

    lp_filter = lowPassFilter(img)

    mean_img = lp_filter.meanFilter()
    ax2.imshow(mean_img, cmap='gray')
    ax2.set_title('Mean Filter')

    gauss_img = lp_filter.GaussFilter()
    ax3.imshow(gauss_img, cmap='gray')
    ax3.set_title('Gaussian Filter')

    plt.show()
