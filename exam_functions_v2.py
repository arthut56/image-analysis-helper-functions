import glob
import math
from skimage.filters import *
from skimage.color import rgb2gray
from skimage import color, io, measure, filters
from skimage.measure import profile_line
from skimage.transform import *
from skimage.util import img_as_float, img_as_ubyte
from sklearn import decomposition
from scipy.spatial import distance
import numpy as np
import pydicom as dicom
from scipy.ndimage import correlate
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.morphology import erosion, dilation, opening, closing, footprint_rectangle
from skimage.morphology import disk
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
import sympy as sp


#Class 1 Basics


def read_image_from_path(path):
    image = io.imread(path)
    print("Image dimensions: " + str(image.shape))
    print("With pixel ranges: " + str(np.min(image)) + ", " + str(np.max(image)))
    return image

def read_dcom_from_path(path):
    #to get pixel values do .pixelarray
    image = dicom.dcmread(path).pixel_array
    print("Image dimensions: " + str(image.shape))
    print("With pixel ranges: " + str(np.min(image))) + ", " + str(np.max(image))
    return image

def read_txt(path, delimiter=None):
    return np.loadtxt(path, comments="%", delimiter=delimiter)


def get_dimensions(image):
    return image.shape

def get_pixel_type(image):
    return image.dtype

"""
Images are displayed with:
plt.imshow(image)
plt.show(image)
"""


"""
Change image to unsinged bytes:
img_as_ubyte(image)

Change image to float:
img_as_float(img_in)

Change image to grayscale:
rgb2gray(img_in)

Change image to HSV:
rgb2hsv(img_in)

"""



###############################
#PCA Analysis
# Manual

def standardize_data(data, axis=0):
    return (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)


"""
Data is standardized
cov_matrix = np.cov(data.T)
values, vectors = np.linalg.eig(cov_matrix)
variation_norm = values / values.sum() * 100
projection = vectors.T.dot(data_matrix.T)

To get variation explained by 1st component, do variation_norm[0]


To project data on PCA do:
    projection = vectors.T.dot(data_matrix.T)
    
To extract pca 1 -> projection[0, :]
To extract pca 1-3 -> projection[0:3,:]
    
*Transpose is done when input matrix has features as columns and datapoints as rows

PAIRPLOTS:

pca1to3 = projection[0:3, :]

    d = pd.DataFrame(pca1to3.T)
    sns.pairplot(d)
    plt.show()
    
    
Plot comparison of PCA1, PCA2
plt.plot(projection[0,:], projection[1,:])

"""



"""
AUTOMATIC

Read the data matrix as before, but do not subtract the mean. The procedure subtracts the mean for you.

pca = decomposition.PCA()
pca.fit(data_matrix)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x)


Select PCA1, ...
data_transform[:, 0], data_transform[:, 1]

To project a single image,
im1 = read_image_from_path(...).flatten().reshape(1,-1)
pca.transform(im1)

"""



##################################
##Cameras and Lenses

"""
1/g + 1/b = 1/f

FOV hori = 2* arctan (sensor width/2f)
FOV vert = 2* arctan (sensor height/2f)

"""



##Pixels greater than threshold are white
def threshold_image(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 1
    return img

"""
Thresholds: (img_r > 100) & (img_g < 100)
"""









##################################################
#Geometric Transformations

"""
- Rotation (2D):

resulting_image = rotate(img, 16, center=(20,20))
                               ^ degrees
                               
                               
- Euclidean Transform (Translation + Rotation):

rotation_angle = 10.0 * math.pi / 180.
translation_vector = [10, 20]
transform = EuclideanTransform(rotation=rotation_angle, translation=translation_vector)
                                            ^ radians

This prints the transformation matrix:                               
print(transform.params)


- Similarity Transform (Translation + Rotation + Scaling):

transform = SimilarityTransform(rotation=rotation_angle, translation=translation_vector, scaling=float)
                                        ^ radians
      
                                      
To apply a matrix, you use warp:

warped_im = warp(image, transform.inverse)
                                ^ apply inverse
                                                    
- Swirl:

swirled_im = swirl(image, strength=int, radius=radians)
"""


############################################33
#Landmark based registration (intro)

"""
src = landmark points for source image
dst = landmark points for destination image

landmark error :

e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error F: {f}")


Find optimal translation (simple method):

Find the average pixel in src, dst.
Do av_dst - av_src to find the vector that maps src to dst


Find optimal transformation:

tform = EuclideanTransform()/SimilarityTransform()

tform.estimate(src, dst)

The found transformation can be applied to the source points (to map them to dst) by doing:
source_tform = matrix_transform(src, tform.params)

To apply the source image to the destination you do:

warped_im = warp(src_image, tform.inverse)
                                ^ needs to have estimated with src, dst

"""

#############################################################
#Cats Cats Cats



#synthesization, create new images from existing images
#you need as many weights as PCA components you want

"""
synthesized_im = average_im + w0 * pca0 + w1 * pca1...


euclidean distance:

result = all images projected on pca[components1,2] - your image projected on pca[components1,2]

pca distances = np.linalg.norm(result, axis=1)

"""


###########################################################3
#Hough Space

"""
r = x cos theta + y sin theta

*theta needs to be in radians (np.deg2rad)


When measuring from x,y coordinate system:

Get the length of the line, ro.
Get the angle with respect to the perpendicular line originating from the origin.
"""

############################################################
#Gradient Descent

x1,x2 = sp.symbols('x,y')
fun = 2*x1 -3*x2 + x1*x2


def gradient_descent(x_1_start, x_2_start, func, step_length, n_steps):
    grad_x_1 = sp.diff(func, x1)
    grad_x_2 = sp.diff(func, x2)
    x_1 = x_1_start
    x_2 = x_2_start
    #errors array
    cs = []
    xarr = [x_1]
    yarr = [x_2]
    xvect = np.array([x_1, x_2])
    gradvect = np.array([ grad_x_1 ,grad_x_2  ])
    for i in range(n_steps - 1):
        xvect = xvect - step_length * np.array([gradvect[0].subs(x1, xvect[0]).subs(x2, xvect[1]),
                                              gradvect[1].subs(x1, xvect[0]).subs(x2, xvect[1]) ])
        c = fun.subs(x1, xvect[0]).subs(x2, xvect[1]).evalf()
        xarr.append(xvect[0])
        yarr.append(xvect[1])


        #NUMBER OF ITERATIONS IS I + 1
        #if c < 2.0:
        #    print(i)
        #    break

        x_1 = xvect[0]
        x_2 = xvect[1]

        cs.append(float(c))


    #plots green circles with line -
    plt.plot(xarr, yarr, "go-")
    plt.show()

    plt.plot(cs)
    plt.show()

#############################################
#Gradient descent (with countour)
"""
def gradient_descent_f_2024(x_1_start, x_2_start, func, step_length, n_steps, var_x1, var_x2):
    grad_x_1 = sp.diff(func, var_x1)
    grad_x_2 = sp.diff(func, var_x2)

    x_1 = x_1_start
    x_2 = x_2_start
    cs = []
    path = [(x_1, x_2)]

    for i in range(n_steps):
        grad1_val = grad_x_1.subs({var_x1: x_1, var_x2: x_2}).evalf()
        grad2_val = grad_x_2.subs({var_x1: x_1, var_x2: x_2}).evalf()
        x_1 = float(x_1 - step_length * grad1_val)
        x_2 = float(x_2 - step_length * grad2_val)

        path.append((x_1, x_2))

        c = func.subs({var_x1: x_1, var_x2: x_2}).evalf()
        if c < 2:
            print(i+1)
        cs.append(float(c))

    # Convert path to NumPy array
    path = np.array(path)

    # Plot contour of function
    x_range = np.linspace(-6, 6, 100)
    y_range = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z_func = sp.lambdify((var_x1, var_x2), func, 'numpy')
    Z = Z_func(X, Y)

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)

    # Plot path
    plt.plot(path[:, 0], path[:, 1], 'ro--', label='Gradient Descent Path')
    plt.scatter(path[0, 0], path[0, 1], color='blue', label='Start')
    plt.scatter(path[-1, 0], path[-1, 1], color='green', label='End')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gradient Descent Path on Function Contour')
    plt.legend()
    plt.grid(True)
    plt.show()
"""

#########################################################
#3D image registration

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.util import img_as_ubyte

vol_sitk = sitk.ReadImage(...)


def rotation_matrix(pitch=0, roll=0, yaw=0, deg=False):
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

    #Pitch
    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])

    #Roll
    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    #Yaw
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])



    x,y,z = 10, 0, 0
    Tr = np.array([[1, 0, 0, x],
                    [0,1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])

    R = np.dot(Tr, np.dot(np.dot(R_x, R_y), R_z))


    ##Change as needed
    R = np.dot(np.dot(R_x, R_y), R_z)
    #apply transformation from left to right, first left, then right
    return R



def imshow_orthogonal_view(sitkImage, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = img_as_ubyte(data/np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

# Display the volume
imshow_orthogonal_view(vol_sitk, title='T1.nii')


# Create the Affine transform and set the rotation
transform = sitk.AffineTransform(3)

centre_image = np.array(vol_sitk.GetSize()) / 2 - 0.5 # Image Coordinate System
centre_world = vol_sitk.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System
rot_matrix = rotation_matrix(0, 0, 0)[:3, :3] # SimpleITK inputs the rotation and the translation separately
                #in radians


transform.SetCenter(centre_world) # Set the rotation centre
transform.SetMatrix(rot_matrix.T.flatten())

# Apply the transformation to the image
ImgT1_A = sitk.Resample(vol_sitk, transform)

translation = [0, 5, 0]

transform.SetTranslation(translation)


# Set the registration - Fig. 1 from the Theory Note
R = sitk.ImageRegistrationMethod()

# Set a one-level the pyramid schedule. [Pyramid step]
R.SetShrinkFactorsPerLevel(shrinkFactors = [2])
R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set the interpolator [Interpolation step]
R.SetInterpolator(sitk.sitkLinear)

# Set the similarity metric [Metric step]
R.SetMetricAsMeanSquares()

# Set the sampling strategy [Sampling step]
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.50)

# Set the optimizer [Optimization step]
R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

# Initialize the transformation type to rigid
initTransform = sitk.Euler3DTransform()
R.SetInitialTransform(initTransform, inPlace=False)


# Estimate the registration transformation [metric, optimizer, transform]
tform_reg = R.Execute(fixed_image, moving_image)

# Apply the estimated transformation to the moving image
ImgT1_B = sitk.Resample(moving_image, tform_reg)





#Convenience functions
def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()












