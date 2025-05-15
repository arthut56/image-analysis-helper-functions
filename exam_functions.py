import glob
import math

from skimage.color import rgb2gray
from skimage import color, io, measure, filters
from skimage.measure import profile_line
from skimage.transform import rescale, resize, SimilarityTransform, matrix_transform
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte
from sklearn import decomposition
from scipy.spatial import distance
import numpy as np
import pydicom as dicom
from scipy.ndimage import correlate
from scipy.stats import norm
import seaborn as sns
import pandas as pd


#Class 1 Basics


def read_image_from_path(path):
    return io.imread(path)

def read_dcom_from_path(path):
    #to get full image to .pixelarray
    return dicom.dcmread(path)

def read_txt(path):
    return np.loadtxt(path, comments="%")


def get_dimensions(image):
    return image.shape

def get_pixel_type(image):
    return image.dtype

#Images are displayed with:
#plt.imshow(image)
#plt.show(image)


##Thresholding an image
#

#############################
#PCA

#Receives projection as input, if you want PCA 1 -3, do projection[0:3, :] as input
def make_pairplot_from_components(components):
    d = pd.DataFrame(components.T)
    sns.pairplot(d)


def pca_on_data_datamatrix(data):
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)

    cov_matrix = np.cov(data.T)
    values, vectors = np.linalg.eig(cov_matrix)
    variation_norm = values / values.sum() * 100
    print("Explained Variation between 1 + 2")
    print(variation_norm[0] + variation_norm[1])

    projection = vectors.T.dot(data.T)

    #Data projected on PCA
    """
    To extract pca 1 -> projection[0, :]
    To extract pca 1-3 -> projection[0:3,:]
    """

    return projection


#Path needs to end in /
def make_data_matrix(path):
    all_images = glob.glob(path + ".png")
    n_samples = len(all_images)
    im_org = io.imread(all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels
    data_matrix = np.zeros((n_samples, n_features))
    idx = 0
    for image_file in all_images:
        img = io.imread(image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img
        idx += 1

    #NOTE: Mean is not subtracted
    return data_matrix


def all_pca_data_from_matrix(data_matrix):
    pca = decomposition.PCA()
    pca.fit(data_matrix)
    values_pca = pca.explained_variance_
    exp_var_ratio = pca.explained_variance_ratio_
    vectors_pca = pca.components_
    return values_pca, exp_var_ratio, vectors_pca

def data_transform_from_matrix(data_matrix):
    pca = decomposition.PCA()
    pca.fit(data_matrix)

    #To grab a specific component
    #pca.transform(data_matrix)[:, 0]
    #Grabs component 1   ----------^



    #all data projected onto pca space
    return pca.transform(data_matrix)


def image_transform_from_matrix(data_matrix, image):
    pca = decomposition.PCA()
    pca.fit(data_matrix)
    #image is projected onto pca space
    return pca.transform(image)




#############################
#BLOB Analysis

#Array of where each blob is, each blob has a unique pixel value
def find_blobs_from_image(image):
    return measure.label(image)

def get_blobs_characteristics_from_image(image):
    blobs = find_blobs_from_image(image)
    return measure.regionprops(blobs)
#return array, each element holds the information of its corresponding blob:
#area, perimeter, extent (compactness), coords,


#############################
#Similarities

def similarity_landmark(landmarks1, landmarks2):

    #landmarks1 is the image we want to adjust/move/transform
    #onto landmarks2

    e_x = landmarks1[:, 0] - landmarks2[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = landmarks1[:, 1] - landmarks2[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F (before): {f}")

    tform = SimilarityTransform()
    tform.estimate(landmarks1, landmarks2)
    print(f"Answer: scale {tform.scale:.2f}")

    src_transform = matrix_transform(landmarks1, tform.params)

    e_x = src_transform[:, 0] - landmarks2[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - landmarks2[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F (after): {f}")



#############################
#Image Transforms

def rotation_matrix(pitch=0, roll=0, yaw=0, translate = (0,0,0), deg=False):
    """
    Return the rotation matrix associated with the Euler angles roll, pitch, yaw.

    Parameters
    ----------
    pitch : float
        The rotation angle around the x-axis.
    roll : float
        The rotation angle around the y-axis.
    yaw : float
        The rotation angle around the z-axis.
    deg : bool, optional
        If True, the angles are given in degrees. If False, the angles are given
        in radians. Default: False.
    translate: tuple (x,y,z)
    """
    if deg:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])

    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    x,y,z = translate
    Tr = np.array([[1, 0, 0, x],
                    [0,1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])


    ##Change as needed
    R = np.dot(Tr, np.dot(np.dot(R_x, R_y), R_z))
    #apply transformation from left to right, first left, then right


    return R



#############################
#Image Morphology

"""
    ⨁ Dilation
    ⊖ Erosion
    ∘ Opening = Erosion -> Dilation
    · Closing = Dilation -> Erosion

    example: img = closing(erosion(img,SE1),SE2)
"""

#Histogram
def histogram_stretch(img_in):

    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0

    img_out = ((max_desired - min_desired)/(max_val - min_val)) * (img_float - min_val) + min_desired

    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)

"""
plt.hist(im_s.ravel(), bins=256)
plt.title('Image histogram')
io.show()

io.imshow(im_s)
io.show()
"""

############################################
##Thresholding

def minimum_distance_threshold(img1, img2):
    return (np.mean(img1) + np.mean(img2))/2

"""
Apply 2 thresholds: (img_r > 100) & (img_g < 100)
"""

#Image must be binaries
def dice_score(img1, ground_truth):
    dice = 1 - distance.dice(img1.ravel(), ground_truth.ravel())
    return dice


#Needs image in grayscale
def otsu_method(img):
    otsu = filters.threshold_otsu(img)
    return img > otsu

def filter_prewitt_image(image):
    ub = img_as_ubyte(image)
    return filters.prewitt(ub)


def median_filter(image, footprint_size):
    footprint = np.ones([footprint_size, footprint_size])
    return filters.median(image, footprint)


def gaussian_filter(image, sigma):
    return filters.gaussian(image, sigma)

def mean_filter(image, kernel_size):
    weights = np.ones((kernel_size, kernel_size))
    weights /= np.sum(weights)  # Normalize so it averages
    return correlate(image, weights)

#####################
#Hough space

def get_hough_y(x, ro, theta):

    theta = np.deg2rad(theta)
    return (ro - x*np.cos(theta))/np.sin(theta)

"""
r = x cos theta + y sin theta
"""


def linear_gray_transform(in_img, max_desired, min_desired):
    max_val = np.max(in_img)
    min_val = np.min(in_img)
    img_out = (max_desired - min_desired) / (max_val - min_val) * (in_img - min_val) + min_desired
    return img_out




######################
#LDA


def LDA(X, y):
    """
    Linear Discriminant Analysis.

    A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.
    Assumes equal priors among classes

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    W : array-like of shape (n_classes, n_features+1)
        Weights for making the projection. First column is the constants.

    Last modified: 11/11/22, mcbo@dtu.dk
    """

    # Determine size of input data
    n, m = X.shape
    # Discover and count unique class labels
    class_label = np.unique(y)
    k = len(class_label)

    # Initialize
    n_group = np.zeros((k, 1))  # Group counts
    group_mean = np.zeros((k, m))  # Group sample means
    pooled_cov = np.zeros((m, m))  # Pooled covariance
    W = np.zeros((k, m + 1))  # Model coefficients

    for i in range(k):
        # Establish location and size of each class
        group = np.squeeze(y == class_label[i])
        n_group[i] = np.sum(group.astype(np.double))

        # Calculate group mean vectors
        group_mean[i, :] = np.mean(X[group, :], axis=0)

        # Accumulate pooled covariance information
        pooled_cov = pooled_cov + ((n_group[i] - 1) / (n - k)) * np.cov(X[group, :], rowvar=False)

    # Assign prior probabilities
    prior_prob = n_group / n

    # Loop over classes to calculate linear discriminant coefficients
    for i in range(k):
        # Intermediate calculation for efficiency
        temp = group_mean[i, :][np.newaxis] @ np.linalg.inv(pooled_cov)

        # Constant
        W[i, 0] = -0.5 * temp @ group_mean[i, :].T + np.log(prior_prob[i])

        # Linear
        W[i, 1:] = temp

    return W

"""
w = np.linalg.inv(covariance_matrix) @ (mean2 - mean1)

c = ln(P1/P2) - 0.5(mean2 + mean1).T

w0 = c @ w

y(x belongs to class 2) = x.T @ w + w0
if > 0 then x belongs to class 2 else class 1
"""


def normal_dist_from_arr(x, arr):
    ##You can also use norm.pdf
    return (1/(np.std(arr)*math.sqrt(2*math.pi)))*math.exp(-((x - np.mean(arr))**2)/(2*np.std(arr)**2))
