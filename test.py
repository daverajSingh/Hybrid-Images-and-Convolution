import numpy as np
import MyConvolution as myC
from PIL import Image
import cv2

img = cv2.imread('C:/Users/daver/PycharmProjects/COMP3204Handin1/dog.jpg')
cv2.imshow("image 1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
newImg = myC.convolve(img, kernel)
newImg = Image.fromarray(newImg)
cv2.imshow("image 2", newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


