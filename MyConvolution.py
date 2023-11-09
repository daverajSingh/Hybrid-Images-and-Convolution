import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders

    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray  """

    # Your code here. You'll need to vectorise your implementation to ensure it runs  # at a reasonable speed.

    targetRowSize, targetColSize = calculateImageSize(image.shape, kernel.shape)
    kRows, kCols = kernel.shape[0], kernel.shape[1]
    invertedKernel = invertMat(kernel)
    convolvedImage = np.zeros(shape=(targetRowSize,targetColSize))

    for i in range(targetRowSize):
        for j in range(targetColSize):
            mat = image[i:i+kRows, j:j+kCols]
            convolvedImage[i, j] = np.sum(np.multiply(mat, invertedKernel))

    paddingWidth = kCols//2
    paddingHeight = kRows//2

    return addPadding(convolvedImage, paddingWidth, paddingHeight)


def calculateImageSize(imgShape, kShape) -> (int, int):
    imgRows, imgCols = imgShape[0], imgShape[1]
    kRows, kCols = kShape[0], kShape[1]

    numRows = 0
    numCols = 0
    for i in range(imgRows):
        x = i + kRows
        if x <= imgRows:
            numRows +=1

    for i in range(imgCols):
        y = i + kCols
        if y <= imgCols:
            numCols +=1

    return numRows, numCols

def addPadding(image, padW, padH):

    imageWithPadding = np.zeros(shape=(image.shape[0]+ padH*2, image.shape[1] + padW*2))

    imageWithPadding[padH:-padH, padW:-padW] = image

    return imageWithPadding

def invertMat(matrix):
    invertedMat = matrix
    maxX, maxY = matrix.shape[0] - 1, matrix.shape[1] - 1

    for a in range(0,maxX):
        for b in range(0,maxY):
            invertedMat[a][b] = matrix[maxX-a][maxY-b]

    return invertedMat