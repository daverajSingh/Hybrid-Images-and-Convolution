import numpy as np
import MyConvolution as myC
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread('C:/Users/daver/PycharmProjects/COMP3204Handin1/hybrid-images/data/dog.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.array([[0.11, 0.11, 0.11], [0.11, 0.11, 0.11], [0.11, 0.11, 0.11]])
plt.imshow(img)
plt.show()
newImg = myC.convolve(img, kernel)
print(newImg.shape)
print(img.shape)
print(newImg)
print(img)
plt.imshow(newImg)
plt.show()
