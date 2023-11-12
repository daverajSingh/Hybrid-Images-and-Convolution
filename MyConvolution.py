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
    isColour = len(image.shape) == 3
    targetH, targetW = calculateImageSize(image.shape, kernel.shape)
    kRows, kCols = kernel.shape[0], kernel.shape[1]
    invertedKernel = invertMat(kernel)
    paddingWidth = kCols // 2
    paddingHeight = kRows // 2

    if(isColour):
        r,g,b = [image[:,:,i] for i in range(3)]
        rConvolved = calcWeightSum(r, targetH, targetW, invertedKernel, kRows, kCols)
        gConvolved = calcWeightSum(g, targetH, targetW, invertedKernel, kRows, kCols)
        bConvolved = calcWeightSum(b, targetH, targetW, invertedKernel, kRows, kCols)

        print(rConvolved)
        convolvedImage = np.zeros([targetH, targetW, 3])

        for i in range(targetH):
            for j in range(targetW):
                convolvedImage[i, j] = [rConvolved[i][j], gConvolved[i][j], bConvolved[i][j]]
    else:
        convolvedImage = calcWeightSum(image, targetH, targetW, invertedKernel)
    return addPadding(convolvedImage, paddingWidth, paddingHeight, isColour)

def calcWeightSum(channel, height, width, kernel, kernelRows, kernelCols) -> np.ndarray:
    convolvedChannel = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            domain = channel[i:i+kernelCols, j:j+kernelRows]
            convolvedChannel[i, j] = np.sum(np.multiply(domain, kernel))
    return convolvedChannel

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

def addPadding(image, padW, padH, isColour) -> np.ndarray:
    if(isColour):
        imageWithPadding = np.zeros(shape=(image.shape[0] + padH * 2, image.shape[1] + padW * 2, 3))
        imageWithPadding[padH:-padH, padW:-padW] = image
    else:
        imageWithPadding = np.zeros(shape=(image.shape[0]+ padH*2, image.shape[1] + padW*2))
        imageWithPadding[padH:-padH, padW:-padW] = image
    return imageWithPadding

def invertMat(matrix) -> np.ndarray:
    invertedMat = matrix
    maxX, maxY = matrix.shape[0] - 1, matrix.shape[1] - 1

    for a in range(0,maxX):
        for b in range(0,maxY):
            invertedMat[a][b] = matrix[maxX-a][maxY-b]

    return invertedMat

