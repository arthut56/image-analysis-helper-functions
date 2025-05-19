import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
#from IPython.display import clear_output
from skimage.util import img_as_ubyte


#Shows the 3 views cross-section
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

#Places 2 slices on top of each other
def overlay_slices(sitkImage0, sitkImage1, origin = None, title=None):
    """
    Overlay the orthogonal views of a two 3D volume from the middle of the volume.
    The two volumes must have the same shape. The first volume is displayed in red,
    the second in green.

    Parameters
    ----------
    sitkImage0 : SimpleITK image
        Image to display in red.
    sitkImage1 : SimpleITK image
        Image to display in green.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    vol0 = sitk.GetArrayFromImage(sitkImage0)
    vol1 = sitk.GetArrayFromImage(sitkImage1)

    if vol0.shape != vol1.shape:
        raise ValueError('The two volumes must have the same shape.')
    if np.min(vol0) < 0 or np.min(vol1) < 0: # Remove negative values - Relevant for the noisy images
        vol0[vol0 < 0] = 0
        vol1[vol1 < 0] = 0
    if origin is None:
        origin = np.array(vol0.shape) // 2

    sh = vol0.shape
    R = img_as_ubyte(vol0/np.max(vol0))
    G = img_as_ubyte(vol1/np.max(vol1))

    vol_rgb = np.zeros(shape=(sh[0], sh[1], sh[2], 3), dtype=np.uint8)
    vol_rgb[:, :, :, 0] = R
    vol_rgb[:, :, :, 1] = G

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_rgb[origin[0], ::-1, ::-1, :])
    axes[0].set_title('Axial')

    axes[1].imshow(vol_rgb[::-1, origin[1], ::-1, :])
    axes[1].set_title('Coronal')

    axes[2].imshow(vol_rgb[::-1, ::-1, origin[2], :])
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)


def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    Args:
        composite_transform (SimpleITK.CompositeTransform): Input composite transform which contains only
                                                            global transformations, possibly nested.
        result_center (tuple,list): The desired center parameter for the resulting affine transformation.
                                    If None, then set to [0,...]. This can be any arbitrary value, as it is
                                    possible to change the transform center without changing the transformation
                                    effect.
    Returns:
        SimpleITK.AffineTransform: Affine transformation that has the same effect as the input composite_transform.

    Source:
        https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb
    """
    # Flatten the copy of the composite transform, so no nested composites.
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        # The TranslationTransform interface is different from other
        # global transformations.
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            # Some global transformations do not have a translation
            # (e.g. ScaleTransform, VersorTransform)
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c

    return sitk.AffineTransform(A.flatten(), t, c)


#To read a 3D image, do

vol_sitk = sitk.ReadImage("directory")


def rotation_matrix(pitch, roll, yaw, deg=False):
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

    R = np.dot(np.dot(R_x, R_y), R_z)

    return R


"""
TRANSFORMING

An important consideration it is that ITK transforms store the resampling transform/backward mapping transform 
(fixed to moving image). And then, internally, it applies the inverse of the transform to the moving image. This means 
that we have to pass the inverse matrix of the one we have defined. This is because the transformation is applied to 
the moving image and not to the fixed image. It is  important to consider when we want to apply the transformation 
to the fixed image.

Note that the inverse or the rotation matrix is the same as the transpose of the rotation matrix, then, 
when we set the rotation matrix: transform.SetMatrix(rot_matrix.T.flatten())

For a more general transformation matrix (no only rotations involved), you should compute the inverse 
matrix: transform.SetMatrix(np.linealg.inv(rot_matrix).flatten())


e.g.

transform = sitk.AffineTransform(3)

##Might be not necessary in the question#####
centre_image = np.array(vol_sitk.GetSize()) / 2 - 0.5 # Image Coordinate System
centre_world = vol_sitk.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System
transform.SetCenter(centre_world) # Set the rotation centre
#############################################

rot_matrix = rotation_matrix(pitch_radians, 0, 0)[:3, :3] # SimpleITK inputs the rotation and the translation separately
transform.SetMatrix(rot_matrix.T.flatten())
transform.SetTranslation((x, y, z))
#Python tip: When you manually should set the translation, you should use a list and
#not a numpy array.

# Apply the transformation to the image
resampled_im = sitk.Resample(vol_sitk, transform)


"""

def homogeneous_matrix_from_transform(transform):
    """Convert a SimpleITK transform to a homogeneous matrix."""
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = np.reshape(np.array(transform.GetMatrix()), (3, 3))
    matrix[:3, 3] = transform.GetTranslation()
    matrix[3, 3] = 1
    return matrix

#Maybe the angles in the rotation matrix need to be given in negative, somtimes that works

def parameters_from_transform(transform):
    params = transform.GetParameters()
    angles = params[:3]
    print("Angles:")
    print(angles)
    trans = params[3:6]
    print("Translations:")
    print(trans)





"""
FIXED VS MOVING IMAGES

fixed_image = sitk.ReadImage(dir_in + 'ImgT1.nii')
moving_image = sitk.ReadImage(dir_in + 'ImgT1_A.nii')


R = sitk.ImageRegistrationMethod()
# Set the optimizer
R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

# Set the sampling strategy
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.20)

# Set the pyramid scheule
R.SetShrinkFactorsPerLevel(shrinkFactors = [2])
R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set the initial transform
R.SetInterpolator(sitk.sitkLinear)
R.SetMetricAsMeanSquares()

#####Change depending on what you have
# Set the initial transform 

#Initialize the transformation type to rigid 
initTransform = sitk.Euler3DTransform()

#rotation center to the center of the image
initTransform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

R.SetInitialTransform(initTransform, inPlace=False)

tform_reg = R.Execute(fixed_image, moving_image)


#To get the new image, you resample it using the received transformation
ImgT1_B = sitk.Resample(moving_image, tform_reg)
imshow_orthogonal_view(ImgT1_B, title='T1_B.nii')
#you can compare the fixed to the moving image (adjusted)
 overlay_slices(fixedImage, ImgT1_B, title='Overlay')
To get parameters:

    params = tform_reg.GetParameters()
    angles = params[:3]
    trans = params[3:6]
    print('Estimated translation: ')
    print(np.round(trans, 3))
    print('Estimated rotation (deg): ')
    print(np.round(np.rad2deg(angles), 3))

"""


"""
COMBINING TRANSFORMS

# Concatenate - The last added transform is applied first
tform_composite = sitk.CompositeTransform(3)

tform_composite.AddTransform(tform_240.GetNthTransform(0)) 
tform_composite.AddTransform(tform_180.GetNthTransform(0))
tform_composite.AddTransform(tform_60.GetNthTransform(0))
tform_composite.AddTransform(tform_0.GetNthTransform(0))
# Transform the composite transform to an affine transform
affine_composite = composite2affine(tform_composite, centre_world)

##########
e.g.:
1. 30 deg roll
2. 10 x translation
3. 10 deg yaw


transform = sitk.CompositeTransform(3)

transform1 = sitk.AffineTransform(3)
transform2 = sitk.AffineTransform(3)
transform3 = sitk.AffineTransform(3)

transform1.SetMatrix(rotation_matrix(0, -30, 0, deg=True)[:3,:3].T.flatten())
transform2.SetTranslation([10,0,0])
transform3.SetMatrix(rotation_matrix(0, 0, -10, deg=True)[:3,:3].T.flatten())

transform.AddTransform(transform3)
transform.AddTransform(transform2)
transform.AddTransform(transform1)

transform = composite2affine(transform)
print(homogeneous_matrix_from_transform(transform))
######

Or alternative:

R1 = rotation_matrix(0, -30, 0, deg=True)
T = np.array([[1, 0, 0, 10],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
R3 = rotation_matrix(0, 0, -10, deg=True)


First goes to the right
print(R3 @ T @ R1)
"""







"""
3D <-> ARRAY

sitk.GetImageFromArray(im_as_array.astype(np.uint8))
                                ^^^^^^^^^^^^^^^ when binary
                                
sitk.GetArrayFromImage(im_as_3d)


"""

def normalized_correlation(img1, img2):
    """
    Calculate the normalized correlation coefficient between two images.
    Both img1 and img2 should be NumPy arrays of the same shape.
    """
    # Flatten the images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Subtract the mean
    img1_mean_centered = img1_flat - np.mean(img1_flat)
    img2_mean_centered = img2_flat - np.mean(img2_flat)

    # Compute the normalized correlation coefficient
    numerator = np.sum(img1_mean_centered * img2_mean_centered)
    denominator = np.sqrt(np.sum(img1_mean_centered ** 2) * np.sum(img2_mean_centered ** 2))

    if denominator == 0:
        return 0  # Avoid division by zero

    return numerator / denominator

#Mean squared error
def mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)