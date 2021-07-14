from os import WEXITED
import numpy as np
import cv2 as cv
import math

# Algorithm:
# Load in image from storage
# Find ROI
# Slice up ROI
# Once located:
#   Give target dimension range
#   Find maximum dimension (either width or height or even both)
#   Find scale value between target dimension range and maximum dimension
#   Scale and return scaled ROI

# Assume that the image already exists. You may need to modify your directory
# or this line to reflect what is being done in this script
# target_image = "TargetSubset/pngs/26518_A.png"
target_image = "PRE_TEST.png"
flip_blackwhite = True


def LoadImage():
    print("Loading Image")
    img = cv.imread(target_image)

    if flip_blackwhite:
        img = cv.bitwise_not(img)

    if img is None:
        print("Unable to load image.")
        exit(-1)
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    original_dim_height, original_dim_width = img.shape

    # cv.imshow("Example", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return img, original_dim_height, original_dim_width


def FindROI(img):
    # print("Finding Region of interest")
    height, width = img.shape
    # print(height, width)

    # Set both values to some impossible dimesions
    x_min = 1000000
    y_min = 1000000
    
    x_max = -1
    y_max = -1

    # Now we can find the ROI
    for y in range(0, height):
       for x in range(0, width):
           if img[y,x] == 255:
                if x_max < x: 
                    x_max = x
                if y_max < y: 
                    y_max = y
                if x_min > x: 
                    x_min = x
                if y_min > y: 
                    y_min = y

    # print("Dimensions (x_min, y_min) -> ({}, {})".format(x_min,y_min))
    # print("Dimensions (x_max, y_max) -> ({}, {})".format(x_max,y_max))

    # cv.rectangle(img, (x_min,y_min), (x_max, y_max), (255,255,255), 1)

    # cv.imshow("Example", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return (x_min, x_max, y_min, y_max)

def RescaleROI(img, roi, dimension_limit):
    print("Resizing ROI")
    
    x_dim = roi[1] - roi[0]
    y_dim = roi[3] - roi[2]

    # get copy of ROI
    roi = img[roi[2]:roi[3], roi[0]:roi[1]]

    # check if we need to even scale at all
    if x_dim == dimension_limit or y_dim == dimension_limit:
        return roi
    
    dim_limiter = 0

    if x_dim > y_dim:
        dim_limiter = x_dim
    elif x_dim < y_dim:
        dim_limiter = y_dim
    else:
        dim_limiter = x_dim

    scale_value = dimension_limit/dim_limiter

    print("Scale Value: ", scale_value)
    
    new_x = math.floor(x_dim * scale_value)
    new_y = math.floor(y_dim * scale_value)

    print("New dimensions: ({}, {})".format(new_x, new_y))
    
    roi_rescale = cv.resize(roi, (new_y, new_x))

    ret, thresh = cv.threshold(roi_rescale, 80, 255, cv.THRESH_BINARY)

    # cv.imshow("Before", roi)
    # cv.imshow("After", thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return thresh

    
def createNewImage(roi, height, width):

    # Create base black image of original dimensions
    base_image = np.zeros((height, width), np.uint8)

    # Now we find the position to place the ROI.
    # first we calculate the half dimension on base_image
    # then calculate the half dimension on ROI.
    # Lastly, we take the half dimension of base - half dimension of ROI.
    half_height = height/2
    half_width = width/2
    
    roi_height, roi_width = roi.shape
    roi_half_height = roi_height / 2
    roi_half_width = roi_width / 2

    wriggle = math.floor(half_height - roi_half_height)
    margin = math.floor(math.sqrt(height) + math.sqrt(wriggle)) # Something that scale with image fairly well.
    
    position_height = math.floor(half_height-roi_half_height)
    position_width = math.floor(half_width-roi_half_width)

    # print(half_height, half_width)
    # print(roi_height, roi_width, roi_half_height, roi_half_width)
    # print(position_height, position_width)
    # print("Radius of Base image = ", half_height)
    # print("Radius of ROI image = ", roi_half_height)
    # print("Wriggle room of ROI in Base image = ", wriggle)
    # print("Margin room of ROI in Base image = ", margin)

    locations = [ \
        (position_height, position_width), # Center
        (position_height - wriggle + margin, position_width), # Top
        (position_height + wriggle - margin, position_width), # Bottom
        (position_height, position_width - wriggle + margin), # Left
        (position_height, position_width + wriggle - margin) # Right
    ]

    for locale in locations:
        translocateROI(roi.copy(), base_image.copy(), locale[0], locale[1])
        
    

def translocateROI(roi, base_image, center_height, center_width):
    roi_height, roi_width = roi.shape
    # print(center_height, center_width)
    base_image[center_height:roi_height+center_height, center_width:roi_width+center_width] = roi
    cv.imshow("Empty_"+str(roi.shape)+"_"+str(center_height) + "_" + str(center_width), base_image)

if __name__ == "__main__":
    print("Hello world!")
    img, height, width = LoadImage()
    cv.imshow("Original", img)
    roi_dim = FindROI(img)

    # roi = RescaleROI(img, roi_dim, 100)
    # createNewImage(roi, height, width)
    # roi = RescaleROI(img, roi_dim, 175)
    # createNewImage(roi, height, width)
    # roi = RescaleROI(img, roi_dim, 250)
    # createNewImage(roi, height, width)
    roi = RescaleROI(img, roi_dim, 300)
    createNewImage(roi, height, width)

    cv.waitKey(0)
    cv.destroyAllWindows()