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

    # Checks if the picture is coloured or greyscale
    isColour = len(image.shape) == 3

    # Sizes of Image and Kernel
    imageHeight, imageWidth = image.shape[0], image.shape[1]
    kHeight, kWidth = kernel.shape[0], kernel.shape[1]

    # Flip Kernel
    kernel = invertMat(kernel)

    # Padding Dimensions
    paddingHeight = kHeight // 2
    paddingWidth = kWidth // 2

    # Convert to floats between 0 and 1
    image = (image/255).astype(np.float64)

    # Picture dimensions in the padded image
    imageEdgeX = paddingWidth + imageWidth
    imageEdgeY = paddingHeight + imageHeight

    # Create Zero-Padded Image
    if isColour:
        paddedImage = np.zeros((imageHeight + 2*paddingHeight, imageWidth + 2*paddingWidth, 3), dtype=np.float64)
        paddedImage[paddingHeight:imageEdgeY, paddingWidth:imageEdgeX] = image

        # Init Convolved Image
        convolvedImage = np.zeros((imageHeight, imageWidth, 3), dtype=np.integer)

    else:
        paddedImage = np.zeros((imageHeight + 2 * paddingHeight, imageWidth + 2 * paddingWidth), dtype=np.float64)
        paddedImage[paddingHeight:imageEdgeY, paddingWidth:imageEdgeX] = image

        # Init Convolved Image
        convolvedImage = np.zeros((imageHeight, imageWidth), dtype=np.integer)


    if isColour:
        for ch in range(3): # Iterates through each channel
            for i in range(imageHeight):
                for j in range(imageWidth):
                    domain = paddedImage[i:i+kWidth, j:j+kHeight, ch]
                    weightedSum = np.sum(np.multiply(domain, kernel))

                    if weightedSum > 1:
                        weightedSum = 1
                    elif weightedSum < 0:
                        weightedSum = 0

                    convolvedImage[i, j, ch] = weightedSum*255
    else:
        for i in range(imageHeight):
            for j in range(imageWidth):
                domain = paddedImage[i:i + kWidth, j:j + kHeight]
                weightedSum = np.sum(np.multiply(domain, kernel))

                if weightedSum > 1:
                    weightedSum = 1
                elif weightedSum < 0:
                    weightedSum = 0

                convolvedImage[i,j] = weightedSum*255
    return convolvedImage

def invertMat(matrix) -> np.ndarray: # Flips matrix
    invertedMat = matrix
    maxX, maxY = matrix.shape[0] - 1, matrix.shape[1] - 1

    for a in range(0, maxX):
        for b in range(0, maxY):
            invertedMat[a][b] = matrix[maxX-a][maxY-b]

    return invertedMat

