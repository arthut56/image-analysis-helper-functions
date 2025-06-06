import numpy as np
from sklearn import decomposition
import glob
from skimage import io


###############################
# PCA Analysis


def standardize_data(data, axis=0):
    return (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)


"""
Careful when using np.mean, min, max, etc:
axis should be 0 when rows are entries
axis should be 1 when columns are entries

MANUAL

Data is standardized beforehand
cov_matrix = np.cov(data.T)
values, vectors = np.linalg.eig(cov_matrix)
variation_norm = values / values.sum() * 100
projection = vectors.T.dot(data_matrix.T)

To get variation explained by 1st component, do variation_norm[0]


To project data on PCA do:
    projection = vectors.T.dot(data_matrix.T)

Select PCA1, ...
projection[0, :], projection[1, :]

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

Careful when using np.mean, min, max, etc:
axis should be 0 when rows are entries
axis should be 1 when columns are entries

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

To get euclidean distances between vectors, do
np.linalg.norm (v2 - v1) #careful with axis


When asked to do pairwise comparisons of several images, e.g.:


    #maps position in data matrix to flower number
    mapping = [8, 15, 10, 9, 14, 2, 4, 12, 1, 5, 6, 13, 7, 3, 11]

    pca1 = data_transform[:, 0]

    pairwise = [(1, 12), (10, 12), (2, 5), (8, 9), (3, 7)]


    for pair in pairwise:
        print(pair)
        print(np.linalg.norm(pca1[mapping.index(pair[0])] - pca1[mapping.index(pair[1])]))
        print("###")


"""


#Path needs to end in /
#reads all image in a folder and places them in data matrix
def make_data_matrix(path, file_termination="*.png"):
    #Change .png to whichever file type you have
    all_images = glob.glob(path + file_termination)
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

    #NOTE: Mean is not subtracted from data_matrix
    return data_matrix, all_images


#When rows are entries
def make_mean_image(data_matrix, axis=0):
    return np.mean(data_matrix, axis=axis)

#Mean squared error
def mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

#Image is flattened in situ
#Sum of squared differences from one image to all images in dataset
def ssd(data_matrix, image):
    #outputs a vector
    return np.sum(np.square(data_matrix - image.flatten()), axis=1)


