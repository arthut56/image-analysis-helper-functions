import numpy as np
from skimage.filters import *
from skimage.util import img_as_float, img_as_ubyte


##Applies threshold automatically
def otsu_method(img):
    otsu = threshold_otsu(img)
    return img > otsu


"""
To threshold an image, you apply it an array of boolean values (a mask).


#All pixel values > 50 become 1, otherwise 0
image_t = image[image > 50]

To apply several thresholds

"""

#Types of filtering

"""
Correlating two images (without border handling):

kernel is e.g: [[0, 1, 0],[1, 2, 1],[0, 1, 0]]

resulting_image = correlate(image, kernel_with_weights)


Correlating two images (with border handling):

res_img = correlate(input_img, weights, mode="constant", cval=10 )

outside border pixels are set to 10
mode can also be "reflect" to copy the bordering pixels into the outside

"""


#You use correlate with this kernel
def mean_filter_kernel(size):
    weights = np.ones([size, size])
    return weights / np.sum(weights)


#Correlation is automatic
def median_filtering(image, kernel_size):
    footprint = np.ones([kernel_size, kernel_size])
    return median(image, footprint)

#Correlation is automatic
def gaussian_filter(image, sigma):
    return gaussian(image, sigma)

#Prewitt here
def filter_prewitt_image(image):
    ub = img_as_ubyte(image)
    return prewitt(ub)


def minimum_distance_threshold_from_images(img1, img2):
    return (np.mean(img1) + np.mean(img2))/2

def minimum_distance_threshold_from_arrays(arr1, arr2):
    return (np.mean(arr1) + np.mean(arr2))/2



def threshold_image(im_org, channel):

    r_comp = im_org[:, :, 0]
    g_comp = im_org[:, :, 1]
    b_comp = im_org[:, :, 2]
    if channel == 2:
        segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
                    (b_comp > 180) & (b_comp < 200)
        return segm_blue
    if channel == 1:
        segm_green = (r_comp < 10) & (b_comp > 85) & (b_comp < 105) & \
                    (g_comp > 180) & (g_comp < 200)
        return segm_green

    if channel == 0:
        segm_red = (b_comp < 10) & (g_comp < 105) & \
                    (r_comp > 180) & (r_comp < 200)
        return segm_red

####################################################33
# Image morphology

"""
    ⨁ dilation
    ⊖ erosion
    ∘ opening = erosion + dilation
    · closing = dilation + erosion

Footprints are made like:
disk(3)
footprint_rectangle(4,4)
diamond(4)...

example: img = closing( erosion(img,disk(3)) , disk(3))
"""



