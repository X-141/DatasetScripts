from ImageRoiRescaling import FindROI

import math
import numpy as np
import cv2 as cv
import os
import regex
import shutil

png_loading_location = "NEWDATA/test/"
target_location = "TargetSubset/"
png_target_location = "TargetSubset/pngs"
knn_file = "TargetSubset/kNN_ETL_Subset.opknn"
dict_file = "TargetSubset/knnDictionary.txt"

# We will be loading only 10 characters identified by their
# pronunciation.
target_characters = ["A", "I", "U", "E", "O", "KA", "N", "TA", "ME", "WA"]
total_png_per_char = 200

# Some globals to influence image transformation.
target_dimensions = 32
perform_threshold = True
threshold_value = 80


def trainKNNSubset():
    print("Loading {} target png of characters: {}".format(total_png_per_char, target_characters))

    if not os.path.exists(png_loading_location):
        print("Location to load pngs from does not exist. Exiting")
        exit(-1)

    if not os.path.exists(target_location):
        os.mkdir(target_location)
        os.mkdir(png_target_location)

    for character in target_characters:
        re = regex.compile(r"\d*_{}.png".format(character))
        current_total = 0
        for pngs in os.listdir(png_loading_location):
            if re.fullmatch(pngs) is not None:
                # print(os.path.join(os.path.abspath(png_loading_location), pngs))
                shutil.copy(os.path.join(os.path.abspath(png_loading_location), pngs), \
                    png_target_location)
                current_total = current_total + 1
                if current_total == total_png_per_char:
                    break

    trainKnnModel_ETL()


def trainKnnModel_ETL():
    print("Executing training ETL model.")

    data_set = os.listdir(png_target_location)
    dataset_size = len(data_set)
    x_train = np.empty(shape=[dataset_size, target_dimensions, target_dimensions])
    y_train = np.empty(shape=[dataset_size, 1])
    re = regex.compile('\d*_(.*).png')

    character_dict = dict()
    character_value = 0
    row_index = 0
    
    for entry in data_set:
        filepath = os.path.join(png_target_location, entry)
        if os.path.exists(filepath):
            img = cv.imread(filepath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # cv.imshow("Before", img)

            img_height, img_width = img.shape
            roi_dim = FindROI(img)
            roi = img[roi_dim[2]:roi_dim[3], roi_dim[0]:roi_dim[1]]
            centered_roi = centerROI(roi, img_height, img_width)
            
            # cv.imshow("After", centered_roi)
            # cv.waitKey()
            # cv.destroyAllWindows()

            target_img = cv.resize(centered_roi, (target_dimensions,target_dimensions), interpolation=cv.INTER_AREA)
            character = re.match(entry).group(1)

            if perform_threshold:
                ret, target_img = cv.threshold(target_img, threshold_value, 255, cv.THRESH_BINARY)
            
            # print(character)
            # cv.imshow("Example", target_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            
            if not (character in character_dict):
                # print(entry)
                character_dict[character] = character_value
                character_value = character_value + 1

            y_train[row_index] = character_dict[character]
            x_train[row_index] = target_img
            row_index = row_index + 1
            

    print(character_dict)

    X = x_train.reshape(len(x_train), -1)
    X = np.float32(X)
    Y = y_train
    Y = np.float32(Y)

    print(X.shape)
    print(Y.shape)

    knn = cv.ml.KNearest_create()
    knn.train(X, cv.ml.ROW_SAMPLE, Y)
    knn.save(knn_file)

    with open(dict_file, "w") as file:
        for key in character_dict:
            file.writelines(key + "," + str(character_dict[key]) + '\n')
        

def centerROI(roi, height, width):
    base_image = np.zeros((height, width), np.uint8)

    half_height = height/2
    half_width = width/2

    roi_height, roi_width = roi.shape
    roi_half_height = roi_height / 2
    roi_half_width = roi_width / 2

    wriggle = math.floor(half_height - roi_half_height)
    margin = math.floor(math.sqrt(height) + math.sqrt(wriggle)) # Something that scale with image fairly well.

    position_height = math.floor(half_height-roi_half_height)
    position_width = math.floor(half_width-roi_half_width)

    base_image[position_height:roi_height+position_height, position_width:roi_width+position_width] = roi

    return base_image

if __name__ == "__main__":
    print("Hello!")
    trainKNNSubset()
