import numpy as np
import math
from MyConvolution import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float

    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float

    :returns returns the hybrid image created by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
        a high-pass image created by subtracting highImage from highImage convolved with
        a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """
    # Assumes both images are either colour or greyscale
    isColour = len(lowImage.shape) == 3

    # Generates Low Pass Image
    lowPassImage = getLowImage(lowSigma, lowImage)

    # Generates High Pass Image
    highpassImage = getHighImage(highSigma, highImage)

    # Combines both images to get Hybrid
    hybridImage = lowPassImage + highpassImage


    # Clip values in between 0 and 255
    for x in range(len(hybridImage)):
        for y in range(len(hybridImage[0])):
            if isColour:
                for z in range(len(hybridImage[0][0])):
                    hybridImage[x][y][z] = max(min(hybridImage[x][y][z], 255), 0)
            else:
                hybridImage[x][y] = max(min(hybridImage[x][y], 255), 0)

    return hybridImage.astype(np.uint8)


def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    # Calculate size as to spec
    size = np.floor(8*sigma+1)
    if size % 2 == 0:
        size += 1

    size = int(size)

    center = size // 2
    kernel = np.zeros((size, size))

    # Generate Gaussian
    for y in range(size):
        for x in range(size):
            diff = (y - center) ** 2 + (x - center) ** 2
            kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

    # Normalise Kernel
    kernel = kernel / np.sum(kernel)

    return kernel

def getLowImage(lowSigma, lowImage) -> np.ndarray:
    return convolve(lowImage, makeGaussianKernel(lowSigma))

def getHighImage(highSigma, highImage) -> np.ndarray:
    return (highImage - getLowImage(highSigma, highImage))
