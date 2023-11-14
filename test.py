import numpy as np
import MyConvolution as c
import MyHybridImages as h
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread('C:/Users/daver/PycharmProjects/COMP3204Handin1/Donald_Trump_official_portrait (1) (1).jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('C:/Users/daver/PycharmProjects/COMP3204Handin1/Vladimir_Putin_-_2012.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

def test1(a, b):
    lowImage = h.getLowImage(a, img)
    plt.imshow(lowImage)
    plt.show()

    highImage = h.getHighImage(b, img2)
    plt.imshow(highImage)
    plt.show()

    hyb = h.myHybridImages(img, a, img2, b)
    plt.imshow(hyb)
    plt.show()
    plt.imsave('save.png', hyb)

def test2(img):
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    imgC = c.convolve(img,sharpen)
    plt.imshow(img)
    plt.show()
    plt.imshow(imgC)
    plt.show()
    imgcv2 = cv2.filter2D(img, -1, sharpen, borderType=cv2.BORDER_CONSTANT)
    plt.imshow(imgcv2)
    plt.show()
    comp = imgC == imgcv2
    # Count the number of True values
    num_true = np.count_nonzero(comp)

    # Count the number of False values
    num_false = np.prod(comp.shape) - num_true

    print(num_true / (num_false + num_true))
    print(img.shape)
    print(imgC.shape)

def hybrid_images(image1, image2, sigma_low, sigma_high):
    # Create low-pass filter kernel
    kernel_low = h.makeGaussianKernel(sigma_low)

    # Apply low-pass filter to image1
    image1_low = cv2.filter2D(image1, -1, kernel_low, borderType=cv2.BORDER_CONSTANT)

    # Create high-pass filter kernel
    kernel_high = h.makeGaussianKernel(sigma_high)

    # Apply high-pass filter to image2
    image2_high = image2 - cv2.filter2D(image2, -1, kernel_high, borderType=cv2.BORDER_CONSTANT)

    # Create hybrid image by combining low-pass and high-pass components
    hybrid_image = image1_low + image2_high

    return hybrid_image

test1(3,15)