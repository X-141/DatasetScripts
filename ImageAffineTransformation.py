import numpy as np
import cv2 as cv
import math

import ImageProcessingMethods as imgproc

# This script will do the following:
# Rescale the image by 85% and 115%
# Tilt the image Left by 15 degrees
# Tilt the image Right by 15 degrees

target_image = "Sample_A.png"

# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
def NewROIScalar(img, roi, scalar):
    print("Resizing ROI")

    x_dim = roi[1] - roi[0]
    y_dim = roi[3] - roi[2]

    # get copy of ROI
    roi = img[roi[2]:roi[3], roi[0]:roi[1]]

    new_x_dim = math.floor(x_dim * scalar)
    new_y_dim = math.floor(y_dim * scalar)

    if new_x_dim > img.shape[0] or new_y_dim > img.shape[1]:
        print("Scalar size indicated would exceed base image dimension.")
        return

    print("New dimensions: ({}, {})".format(new_x_dim, new_y_dim))
    
    roi_rescale = cv.resize(roi, (new_y_dim, new_x_dim))

    ret, thresh = cv.threshold(roi_rescale, 80, 255, cv.THRESH_BINARY)

    cv.imshow("Before", roi)
    cv.imshow("After", thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return thresh    

# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
# by: nicodjimenez
def ApplyLeftAndRightTilt(img, roi):
    roi_center = tuple(np.array([roi[3],roi[1]])/2)
    rot_mat_left = cv.getRotationMatrix2D(roi_center, 15.0, 1.0)
    rot_mat_right = cv.getRotationMatrix2D(roi_center, -15.0, 1.0)

    roi_cpy_1 = img[roi[2]:roi[3], roi[0]:roi[1]]
    roi_cpy_2 = img[roi[2]:roi[3], roi[0]:roi[1]]

    left_rotation = cv.warpAffine(roi_cpy_1, rot_mat_left, (roi[3],roi[1]))
    right_rotation = cv.warpAffine(roi_cpy_2, rot_mat_right, (roi[3],roi[1]))

    ret, left_rotation = cv.threshold(left_rotation, 80, 255, cv.THRESH_BINARY)
    ret, right_rotation = cv.threshold(right_rotation, 80, 255, cv.THRESH_BINARY)

    cv.imshow("Left Rotation", left_rotation)
    cv.resizeWindow('Left Rotation', 200, 200)
    #cv.imshow("Right Rotation", right_rotation)
    #cv.resizeWindow('Right Rotation', 200, 200)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return (left_rotation, right_rotation)

def createNewImage(roi, roi_height, roi_width, height, width):
    base_image = np.zeros((height, width), np.uint8)

    half_height = height/2
    half_width = width/2
    
    roi_half_height = roi_height / 2
    roi_half_width = roi_width / 2

    position_height = math.floor(half_height-roi_half_height)
    position_width = math.floor(half_width-roi_half_width)

    newTranslocateROI(roi, base_image, roi_height, roi_width, position_height, position_width)

def newTranslocateROI(roi, base_image, roi_height, roi_width, center_height, center_width):
    base_image[center_height:roi_height+center_height, center_width:roi_width+center_width] = roi
    cv.imshow("Empty_"+str(roi.shape)+"_"+str(center_height) + "_" + str(center_width), base_image)

if __name__ == "__main__":
    print("Hello!")
    img, height, width = imgproc.LoadImage(target_image)
    roi_dim = imgproc.FindROI(img)
    #roi_small = NewROIScalar(img, roi_dim, .85)
    #roi_large = NewROIScalar(img, roi_dim, 115)
    
    left_rot, right_rot = ApplyLeftAndRightTilt(img, roi_dim)
    new_left_roi = imgproc.FindROI(left_rot)
    left_roi_height = new_left_roi[3] - new_left_roi[2]
    left_roi_width = new_left_roi[1] - new_left_roi[0]
    new_left_rot_roi = createNewImage(left_rot[new_left_roi[2]:new_left_roi[3], new_left_roi[0]:new_left_roi[1]], left_roi_height, left_roi_width, \
        height, width)
    cv.waitKey(0)
    cv.destroyAllWindows()

