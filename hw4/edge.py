import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args -
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns -
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    k_flip = np.flip(kernel, axis=(0,1))
    
    ### YOUR CODE HERE
    
    ### Smoothing the image using a padded array
    for m in range(Hi):
        for n in range (Wi):
            out[m][n] = np.sum(padded[m:m+Hk, n:n+Wk] * k_flip)
   
    pass
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))
    k = (size -1 )// 2
    
    ### YOUR CODE HERE
    
    ### Creating 2-D Gaussian Kernel matrix with the input values
    for x in range(size):
        for y in range(size):
            kernel[x][y] = 1/(2 * np.pi * sigma**2)* np.exp(-1* ((x-k)**2 + (y-k)**2)/(2*sigma**2) )


    pass
    ### END YOUR CODE

    return kernel

def partial_x(image):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None
    
    ### YOUR CODE HERE
    
    ### Vector as a row 
    K = np.array([
        [1/2,0,-1/2],
    ])
    
    pass
    ### END YOUR CODE

    return conv(image, K)

def partial_y(image):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    
    ###Vector as a column
    K = np.array([
        [1/2],
        [0],
        [-1/2]
    ])
    
    pass
    ### END YOUR CODE

    return conv(image, K)

def gradient(image):
    """ Returns gradient magnitude and direction of input img.

    Args:
        image: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(image.shape)
    G_x = partial_x(image)
    G_y = partial_y(image)
    theta = np.zeros(image.shape)

    ### YOUR CODE HERE 
    
    ### Computing direction and magnitude of the gradients based on the given formulas
    
    G = np.sqrt(np.square(G_x) + np.square(G_y))
    theta = np.arctan2(G_y, G_x)
    theta = (np.rad2deg(theta) + 180) % 360
    
    pass
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    ### Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    
    ### sin(x) at rounded points
    
    hashtable_x = {
        0: 0,
        45: 1,
        90: 1,
        135: 1,
        180: 0,
        45 + 180: -1,
        90 + 180: -1,
        135 + 180: -1,
        360: 0
    }
    
    ### cos(x) at rounded points
    hashtable_y = {
        0: 1,
        45: +1,
        90: +0,
        135: -1,
        180: -1,
        45 + 180: -1,
        90 + 180: 0,
        135 + 180: 1,
        360: 1,
    }


    ### BEGIN YOUR CODE
    
    
    ### Loops to compare the data for every pixel
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            approx = int(theta[i][j])
            p1 = G[i-hashtable_x[approx], j-hashtable_y[approx]]
            p2 = G[i+hashtable_x[approx], j+hashtable_y[approx]]
            
            ### Angle is measured clockwisely, if theta = 90 degree the direction is south.
            
            if not (G[i, j] >= p1 and G[i, j] >= p2):
                out[i, j] = 0
            else:
                out[i, j] = G[i, j]
    
    pass
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array which represents strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    
    Hi, Wi = img.shape
    for i in range(Hi):
        for j in range (Wi):
            px = img[i][j]
            if px > high:
                strong_edges[i][j] = px
            if high > px > low:
                weak_edges[i][j] = px
    
    pass
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).
    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)
    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    ### Make new instances of arguments to leave the original references intact
    
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    
    for x in range(1, H-1):
        for y in range(1, W-1):
            Queue = []
            
            if edges[x][y] != 0:
                Queue.append( (x,y))
            while len(Queue) != 0:
                x_, y_, = Queue.pop()
                for (y_n, x_n) in get_neighbors(y_,x_,W,H):
                    if weak_edges[x_n][y_n] != 0:
                        Queue.append((x_n,y_n))
                        edges[x_n][y_n] = weak_edges[x_n][y_n]
                        weak_edges[x_n][y_n] = 0

    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    theta = np.zeros(img.shape)
    G_x = partial_x(img)
    G_y = partial_y(img)
    
    kernel_result = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel_result)
    
    theta = np.arctan2(G_y, G_x)
    theta = (np.rad2deg(theta) + 180) % 360
    
    G, theta = gradient (smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    summ = strong_edges * 1.0 + weak_edges * 0.5
    edges = link_edges(strong_edges, weak_edges)
    
    pass
    ### END YOUR CODE

    return edges


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordinate
    
    for x in range(len(xs)):
        x_axis = xs [x]
        y_axis = ys [x]
        
        for theta in range(num_thetas):
            rho = diag_len + int(x_axis * cos_t[theta] + y_axis * sin_t[theta])
            accumulator[rho, theta] += 1
    
    pass
    ### END YOUR CODE

    return accumulator, rhos, thetas