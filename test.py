import numpy as np
import MyConvolution as myC
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def showImage(img: np.array):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='grey')
    plt.show()

image = Image.open(r'C:\Users\daver\PycharmProjects\COMP3204Handin1\dog.jpg')
image = ImageOps.grayscale(image)
image = image.resize(size=(1000,1000))
image = np.array(image)

showImage(img=image)
print(image.shape)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
image = myC.convolve(image=image, kernel=kernel)
showImage(img=image)
print(image.shape)


