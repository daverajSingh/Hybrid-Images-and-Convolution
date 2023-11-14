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
        # 3D Array of correct shape
        paddedImage = np.zeros((imageHeight + 2*paddingHeight, imageWidth + 2*paddingWidth, 3), dtype=np.float64)
        paddedImage[paddingHeight:imageEdgeY, paddingWidth:imageEdgeX] = image

        # Init Convolved Image
        convolvedImage = np.zeros((imageHeight, imageWidth, 3), dtype=np.integer)

    else:
        # 2D Array of correct shape
        paddedImage = np.zeros((imageHeight + 2 * paddingHeight, imageWidth + 2 * paddingWidth), dtype=np.float64)
        paddedImage[paddingHeight:imageEdgeY, paddingWidth:imageEdgeX] = image

        # Init Convolved Image
        convolvedImage = np.zeros((imageHeight, imageWidth), dtype=np.integer)


    if isColour:
        for ch in range(3): # Iterates through each channel if colour
            for i in range(imageHeight):
                for j in range(imageWidth):
                    # Gets domain of where kernel is applied
                    domain = paddedImage[i:i+kWidth, j:j+kHeight, ch]

                    # Calculates weighted sum of individual point
                    weightedSum = np.sum(np.multiply(domain, kernel))

                    # Clips values to in between 0 and 1
                    if weightedSum > 1:
                        weightedSum = 1
                    elif weightedSum < 0:
                        weightedSum = 0

                    # Adds to array and multiplies by 255
                    convolvedImage[i, j, ch] = weightedSum*255
    else:
        for i in range(imageHeight):
            for j in range(imageWidth):
                # Gets domain of where kernel is applied
                domain = paddedImage[i:i + kWidth, j:j + kHeight]

                # Calculates weighted sum of individual point
                weightedSum = np.sum(np.multiply(domain, kernel))

                # Clips values to in between 0 and 1
                if weightedSum > 1:
                    weightedSum = 1
                elif weightedSum < 0:
                    weightedSum = 0

                # Adds to array and multiplies by 255
                convolvedImage[i,j] = weightedSum*255
    return convolvedImage

def invertMat(matrix) -> np.ndarray: # Flips matrix in x and y direction
    invertedMat = matrix
    maxX, maxY = matrix.shape[0] - 1, matrix.shape[1] - 1

    for x in range(0, maxX):
        for y in range(0, maxY):
            invertedMat[x][y] = matrix[maxX-x][maxY-y]

    return invertedMat

