import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    out = io.imread(image_path)
    pass
    ### END YOUR CODE


    out = out.astype(np.float64) / 255     ### Converting the image to be between the correct range.
    return out


def dim_image(image: np.array):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    out = np.array([0.5 * x**2 for x in image])

    pass
    ### END YOUR CODE
  
    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    
    out  = color.rgb2gray(image) ###Using rgb2gray function
        
    pass
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    to_remove = None
    if channel == 'R':
        to_remove = 0
    elif channel == "G":
        to_remove = 1
    elif channel == "B":
        to_remove = 2
    else:
        raise ValueError("R, G, or B not given!")
        
    copy = image.copy()
    channel = copy[:,:,to_remove]
    channel = np.zeros(channel.shape)
   
    copy[:,:, to_remove] = channel  ### Print(channel.shape)
    
    pass
    ### END YOUR CODE

    return copy


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None
    
    pass
  
    to_split = None
    if channel == 'L':
        to_split = 0
    elif channel == 'A':
        to_split = 1
    elif channel == 'B':
        to_split = 2
    else:
        raise ValueError("incorrect input, expected L, A, or B")
        
    out = lab[:,:,to_split]
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    mapping = {'H': 0, 'S': 1, 'V': 2}

    out = hsv[:,:,mapping[channel]]
    pass
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    
    image_removed = rgb_exclusion(image1, channel1)
    image_dims = image1.shape       
    image_half = np.split(image_removed, 2, axis=1)
    image_half = image_half[0]   
    
    ###Second image
    
    img2_removed = rgb_exclusion(image2 , channel2) 
    img2_half = np.split(img2_removed, 2, axis=1)
    img2_half = img2_half[1] 
    
    out = np.concatenate((image_half, img2_half), axis=1) 
    pass
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
