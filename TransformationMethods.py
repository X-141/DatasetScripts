import numpy as np
import cv2 as cv
import math

# This script will do the following:
# Rescale the image by 85% and 115%
# Tilt the image Left by 1.0 degrees
# Tilt the image Right by 1.0 degrees

target_image = "Sample_A.png"
flip_blackwhite = False
disable_thresholding = False
threshold_value = 50

target_tilt = 6.0

def LoadImage(aImagePath : str):
    #print("Loading Image")
    img = cv.imread(aImagePath)

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

def translocateROI(roi, base_image, center_height, center_width):
    roi_height, roi_width = roi.shape
    # print(center_height, center_width)
    base_image[center_height:roi_height+center_height, center_width:roi_width+center_width] = roi
    return base_image
    #cv.imshow("Empty_"+str(roi.shape)+"_"+str(center_height) + "_" + str(center_width), base_image)

def RefitIntoOriginalImage(img, roi_img):
    roi_height, roi_width = roi_img.shape

    roi_half_height = roi_height / 2
    roi_half_width = roi_width / 2

    height, width = img.shape
    base_image = np.zeros((height, width), np.uint8)

    half_height = height/2
    half_width = width/2

    position_height = math.floor(half_height-roi_half_height)
    position_width = math.floor(half_width-roi_half_width)

    return translocateROI(roi_img, base_image, position_height, position_width)

# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
def CreateROIScaledImage(img, roi, scalar):
    # print("Resizing ROI")
    # cv.imshow("Resizing ROI Before {}".format(scalar), img)

    x_dim = roi[1] - roi[0]
    y_dim = roi[3] - roi[2]

    # get copy of ROI
    roi = img[roi[2]:roi[3], roi[0]:roi[1]]

    new_x_dim = math.floor(x_dim * scalar)
    new_y_dim = math.floor(y_dim * scalar)

    if new_x_dim > img.shape[0] or new_y_dim > img.shape[1]:
        print("Scalar size indicated would exceed base image dimension.")
        return img

    #print("Original Dimensions: ({}, {})".format(x_dim, y_dim))
    #print("New dimensions: ({}, {})".format(new_x_dim, new_y_dim))
    
    new_roi = cv.resize(roi, (new_x_dim, new_y_dim))
    if not disable_thresholding:
        ret, new_roi = cv.threshold(new_roi, threshold_value, 255, cv.THRESH_BINARY)

    new_img = RefitIntoOriginalImage(img, new_roi)

    # cv.imshow("Resizing ROI After {}".format(scalar), new_img)
    return new_img

def CreateTiltedImage(img, roi_dim, tilt_value):

    # we scale down a bit to keep the roi within image bound.
    scaled_down_image = CreateROIScaledImage(img, roi_dim, .90)
    new_roi_dim = FindROI(scaled_down_image)

    # cv.imshow("CreateTiltedImage Before {}".format(tilt_value), img)
    roi_center = tuple(np.array([new_roi_dim[3],new_roi_dim[1]])/2)
    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    # by: nicodjimenez
    rotation_matrix = cv.getRotationMatrix2D(roi_center, tilt_value, 1.0)

    roi_copy = scaled_down_image[new_roi_dim[2]:new_roi_dim[3], new_roi_dim[0]:new_roi_dim[1]]

    rotated_img = cv.warpAffine(roi_copy, rotation_matrix, (new_roi_dim[3],new_roi_dim[1]))
    if not disable_thresholding:
        ret, rotated_img = cv.threshold(rotated_img, threshold_value, 255, cv.THRESH_BINARY)

    # need to do a quick check to see if rotated image exceeds dimensions
    # cv.imshow("CreateTiltedImage After {}".format(tilt_value), rotated_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    new_roi = FindROI(rotated_img)
    new_roi_copy = rotated_img[new_roi[2]:new_roi[3], new_roi[0]:new_roi[1]]
    
    new_img = RefitIntoOriginalImage(img, new_roi_copy)

    #cv.imshow("CreateTiltedImage After {}".format(tilt_value), new_img)

    return new_img


if __name__ == "__main__":
    img, height, width = LoadImage(target_image)
    roi_dim = FindROI(img)
    roi_small = CreateROIScaledImage(img, roi_dim, .85)
    roi_large = CreateROIScaledImage(img, roi_dim, 1.15)
    
    CreateTiltedImage(img, roi_dim, -target_tilt)
    CreateTiltedImage(img, roi_dim, target_tilt)
    cv.waitKey(0)
    cv.destroyAllWindows()

