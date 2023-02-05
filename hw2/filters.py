import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    ### Image matrix rows and columns
    for m in range(Hi):
        for n in range(Wi):
            conv_matrix = 0.0
            ### Kernel matrix rows and columns
            for i in range(Hk):         
                for j in range(Wk):
                    ### The difference betwen the dimensions of both matrices must be more than 0 and less than the image size in order to multiply kernel matrix with the image matrix 
                    if m+1-i < 0 or n+1-j < 0 or m+1-i >= Hi or n+1-j >= Wi:
                        conv_matrix += 0
                    else:
                        conv_matrix += kernel[i][j]*image[m+1-i][n+1-j]
            out[m][n] = conv_matrix
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### Using 'np.zeros'to return an array of given shape and type of zeros using pad and image values
    out = np.zeros((2*pad_height+H, 2*pad_width+W))
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    ### flipping the kernel because of how convolution is defined
    kern = np.flip(kernel, axis=(0,1))
    ima = zero_pad(image, Hk //2, Wk //2)
    out = np.zeros((Hi, Wi))
    
    for m in range(Hi):
        for n in range(Wi):
            ### Using 'np.sum'to make the sum between the kernel and image matrix
            out[m][n] = np.sum(ima[m:m+Hk, n:n+Wk] * kern)
    

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    
    g_ = np.flip(g, axis=(0,1))
    out = conv_fast(f,g_)
    
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    f_mean = np.mean(f, axis=(0,1))
    f_zero_mean = f - f_mean
    g_mean = np.mean(g, axis=(0,1))
    g_zero_mean = g - g_mean
    
    out = cross_correlation(f, g_zero_mean)
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HEREHi, Wi = f.shape
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    
    ### Flipping the kernel because of how convolution is defined
    g_mean = np.mean(g, axis=(0,1))
    g_var = np.var(g, axis=(0,1))

    ker =( g - g_mean) / g_var

    img = zero_pad(f, Hk //2, Wk //2)
    out = np.zeros((Hi, Wi))
    
    for m in range(Hi):
        for n in range(Wi):
            mean_fmn = np.mean(img[m:m+Hk, n:n+Wk], axis=(0,1))
            var_fmn = np.var(img[m:m+Hk, n:n+Wk], axis=(0,1))
            f_patch = (img[m:m+Hk, n:n+Wk] - mean_fmn) / var_fmn
            ### The slice is the same size as the kernel, starting with the zero-padded beginning and ending with the zero-padded end
            out[m][n] = np.sum(f_patch * ker)

    return out
            
    g_mean = np.mean(g, axis=(0,1))
    f_mean = np.mean(f, axis=(0,1))
    g_var = np.var(g, axis=(0,1))
    f_var = np.var(g, axis=(0,1))
    f_ = (f-f_mean)/f_var
    g_ = (g-g_mean)/g_var
    out= cross_correlation(f_, g_)
   
    ### END YOUR CODE

    return out
