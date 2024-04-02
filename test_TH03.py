import matplotlib.pyplot as plt
import cv2
import numpy as np

class test_test_highpass_filter(object):
    def __init__(self,image) -> None:
        self.image = image
    def init_kernel(self, kernel_name:str):
        if kernel_name == "RobertCross":
            return self.show_SharpenedRobert()
        elif kernel_name == "RobertCross1":
            return self.show_RobertCrossGradient1()
        elif kernel_name == "RobertCross2":
            return self.show_RobertCrossGradient2()
        elif kernel_name == "Sobel":
            return self.show_SharpenedSobel()
        elif kernel_name == "SobelX":
            return self.show_SobelX()
        elif kernel_name == "SobelY":
            return self.show_SobelY()
        elif kernel_name == "Laplacian":
            return self.show_SharpenedLaplacian()
        elif kernel_name == "LaplacianV1":
            return self.show_SharpenedLaplacianV1()
        elif kernel_name == "LaplacianV2":
            return self.show_SharpenedLaplacianV2()
        elif kernel_name == "LaplacianV3":
            return self.show_SharpenedLaplacianV3()
        else:
            return self.image
    def Convolution2D(self, kernel):
        m, n = self.image.shape
        img_new = np.zeros([m, n])
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = self.image[i - 1, j - 1] * kernel[0, 0] \
                       + self.image[i, j - 1] * kernel[0, 1] \
                       + self.image[i + 1, j - 1] * kernel[0, 2] \
                       + self.image[i - 1, j] * kernel[1, 0] \
                       + self.image[i, j] * kernel[1, 1] \
                       + self.image[i + 1, j] * kernel[1, 2] \
                       + self.image[i - 1, j + 1] * kernel[2, 0] \
                       + self.image[i, j + 1] * kernel[2, 1] \
                       + self.image[i + 1, j + 1] * kernel[2, 2]
                img_new[i, j] = temp
        img_new = img_new.astype(np.uint8)
        return img_new
    def show_RobertCrossGradient1(self):
        G_cross1 = np.array(([0, 0, 0], [0, -1, 0], [0, 0, 1]), dtype="float")
        return self.Convolution2D(G_cross1)
    def show_RobertCrossGradient2(self):
        G_cross2 = np.array(([0, 0, 0], [0, 0, -1], [0, 1, 0]), dtype="float")
        return self.Convolution2D(G_cross2)
    def show_SharpenedRobert(self):
        return self.show_RobertCrossGradient1() + self.show_RobertCrossGradient2() + self.image
    def show_SobelX(self):
        SobelX = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype="float")
        return self.Convolution2D(SobelX)
    def show_SobelY(self):
        SobelY = np.array(([-1, 0, 1], [-2, 0, 2], [1, 0, 1]), dtype="float")
        return self.Convolution2D(SobelY)
    def show_SharpenedSobel(self):
        return self.show_SobelX() + self.show_SobelY() + self.image
    def show_SharpenedLaplacian(self):
        Laplacian_kerner = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="float")
        return self.image - self.Convolution2D(Laplacian_kerner)
    def show_SharpenedLaplacianV1(self):
        Laplacian_kerner = np.array(([1, 1, 1], [1, -8, 1], [1, 1, 1]), dtype="float")
        return self.image - self.Convolution2D(Laplacian_kerner)
    def show_SharpenedLaplacianV2(self):
        Laplacian_kerner = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]), dtype="float")
        return self.image + self.Convolution2D(Laplacian_kerner)
    def show_SharpenedLaplacianV3(self):
        Laplacian_kerner = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype="float")
        return self.image + self.Convolution2D(Laplacian_kerner)
if __name__ =='__main__':
    image = cv2.imread('/Users/nguyennam/Workspace/XuLyAnh/images/trex.png', 0)
    fig=plt.figure(figsize=(9, 9))
    ax=fig.subplots(2,2)  # hien thi anh goc
    ax[0,0].set_title("Orignal")
    ax[0,0].imshow(image,cmap='gray')
    ## Su dung bo loc RobertCrossGradient
    # hien thi anh loc Robert theo huong thu 1
    Robertcross1=test_test_highpass_filter(image).init_kernel("Sobel")
    ax[0,1].imshow(Robertcross1,cmap='gray')
    ax[0,1].set_title("Anh loc theo huong thu 1")
    # hien thi anh loc Robert theo huong thu 2
    Robertcross2=test_test_highpass_filter(image).init_kernel("SobelX")
    ax[1,1].imshow(Robertcross2,cmap='gray')
    ax[1,1].set_title("Anh loc theo huong thu 2")
    # Ket qua anh loc sac net bang bo loc Robert
    Robertcross=test_test_highpass_filter(image).init_kernel("SobelY")
    ax[1,0].imshow(Robertcross,cmap='gray')
    ax[1,0].set_title("Anh cai thien bang bo loc Sobel")
    plt.savefig("filter_Robert.pdf",bbox_inches='tight')
    plt.show()
