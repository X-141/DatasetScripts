import numpy as np
import cv2 as cv
import math

target_image = "Sample_A.png"
flip_blackwhite = False
disable_thresholding = False

# This threshold provides good results.
threshold_value = 130
# Decent denoising value to remove static.
h_value = 40
target_tilt = 6.0


def denoiseImage(img):
    return cv.fastNlMeansDenoising(img, None, h=h_value)

def thresholdBasic(img):
    # return cv.adaptiveThreshold(img, 255,
    #                                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv.THRESH_BINARY, 23, -2)
    ret, target_img = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY)
    return target_img

def thresholdOtsu(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    ret, target_img = cv.threshold(blur, 0, 255 , cv.THRESH_BINARY+cv.THRESH_OTSU)
    return target_img

def LoadImage(aImagePath : str):
    #print("Loading Image")
    img = cv.imread(aImagePath)

    if flip_blackwhite:
        img = cv.bitwise_not(img)

    if img is None:
        print("Unable to load image.")
        exit(-1)
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img

def FindROI(img):
    height, width = img.shape

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

    return (x_min, x_max, y_min, y_max)

def translocateROI(roi, base_image, center_height, center_width):
    roi_height, roi_width = roi.shape
    # print(center_height, center_width)
    try:
        base_image[center_height:roi_height+center_height, center_width:roi_width+center_width] = roi
    except ValueError as ex:
        raise ex
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

    try:
        return translocateROI(roi_img, base_image, position_height, position_width)
    except ValueError as ex:
        raise ex

# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
def CreateROIScaledImage(img, scalar):

    height, width = img.shape

    new_x_dim = math.floor(width * scalar)
    new_y_dim = math.floor(height * scalar)

    resized_img = cv.resize(img, (new_x_dim, new_y_dim))
    new_img = thresholdOtsu(resized_img)

    roi_dim = FindROI(new_img)
    
    try:
        new_img = RefitIntoOriginalImage(img, new_img[roi_dim[2]:roi_dim[3], roi_dim[0]:roi_dim[1]])
    except ValueError as ex:
        # cv.imshow("Original {}".format(scalar), img)
        # cv.imshow("resized {}".format(scalar), resized_img)
        # cv.imshow("Faulty {}".format(scalar), new_img)
        new_img = thresholdBasic(resized_img)
        #cv.imshow("Resolved {}".format(scalar), new_img)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        print(ex)
        print("\tResolving with harsher thresholding.")
        roi_dim = FindROI(new_img)
        new_img = RefitIntoOriginalImage(img, new_img[roi_dim[2]:roi_dim[3], roi_dim[0]:roi_dim[1]])

    #print("Original Dimensions: ({}, {})".format(width, height))
    #print("New dimensions: ({}, {})".format(new_x_dim, new_y_dim))
    
    #resized_img = cv.resize(img, (new_x_dim, new_y_dim))
    #denoised_img = denoiseImage(resized_img)
    #new_img = thresholdImage(denoised_img)

    #cv.imshow("Before {}".format(scalar),img)
    #cv.imshow("Result {}".format(scalar),new_img)

    #cv.waitKey(0)
    #cv.destroyAllWindows()

    #cv.imshow("Resizing ROI After {}".format(scalar), new_img)

    return new_img


def CreateTiltedImage(img, tilt_value):
    image_center = tuple(np.array(img.shape)/2)

    #rotation_matrix = cv.getRotationMatrix2D(image_center, tilt_value, 1.0)
    #rotated_img = cv.warpAffine(img, rotation_matrix, img.shape)
    #denoise_img = denoiseImage(rotated_img)
    #new_img = thresholdImage(denoise_img)

    rotation_matrix = cv.getRotationMatrix2D(image_center, tilt_value, 1.0)
    rotated_img = cv.warpAffine(img, rotation_matrix, img.shape)
    new_img = thresholdOtsu(rotated_img)

    try:
        roi_dim = FindROI(new_img)
        new_img = RefitIntoOriginalImage(img, new_img[roi_dim[2]:roi_dim[3], roi_dim[0]:roi_dim[1]])
    except ValueError:
        # Backup method to use harsher thresholding.
        new_img = thresholdBasic(rotated_img)
        roi_dim = FindROI(new_img)
        new_img = RefitIntoOriginalImage(img, new_img[roi_dim[2]:roi_dim[3], roi_dim[0]:roi_dim[1]])

    #cv.imshow("Before {}".format(tilt_value), img)
    #cv.imshow("After {}".format(tilt_value), new_img)
    #cv.imshow("Tilting ROI with {}".format(tilt_value), new_img)

    return new_img



if __name__ == "__main__":
    img = LoadImage("11_A.png")
    
    CreateROIScaledImage(img, .85)
    CreateROIScaledImage(img, 1.4)
    
    #CreateTiltedImage(img, -target_tilt)
    #CreateTiltedImage(img, target_tilt)

    cv.waitKey(0)
    cv.destroyAllWindows()

