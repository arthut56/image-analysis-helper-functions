import math
from skimage import measure, segmentation


##########################################################
##BLOB Analysis

#Array of where each blob is, each blob has a unique pixel value
#Image needs to be binary
def find_blobs_from_image(image):
    labels = measure.label(image)
    print(f"Number of blobs: {labels.max()}")
    return labels

#Image needs to be binary
def get_blobs_characteristics_from_image(image):
    blobs = measure.label(image)
    return measure.regionprops(blobs)
#return array, each element holds the information of its corresponding blob:
#area, perimeter, extent (compactness), coords,

"""
areas = np.array([prop.area for prop in region_props])

Label: 1
Coordinates:
[[0 1]
 [0 2]
 [1 1]]

"""

#Image needs to be binary
def remove_border_blobs(image):
    return segmentation.clear_border(image)


def remove_blob_using_coordinates(image, coords):
    for coord in coords:
        x, y = coord
        image[x, y] = 0

    return image


"""
label2rgb shows visualization of the found blobs
labels, image

BLOB Classification by area

min_area =
max_area =

# Create a copy of the label_img
label_img_filter = label_img
for region in region_props:
	# Find the areas that do not fit our criteria
	if region.area > max_area or region.area < min_area:
		# set the pixels in the invalid areas to background
		for cords in region.coords:
			label_img_filter[cords[0], cords[1]] = 0
# Create binary image from the filtered label image
i_area = label_img_filter > 0
show_comparison(img_small, i_area, 'Found nuclei based on area')

"""

def measure_blob_circularity(area, perim):
    est_perim = 2*math.sqrt(math.pi * area)
    return perim / est_perim


"""
Confusion matrix

accuracy = (TP + TN) / N

N = TN + TP + FN + FP

sensitivity = TP / (TP + FN)

specificity = TN / (TN + FP)

FPR = FP / (FP + TN)

TPR = TP / (TP + FN)


"""
