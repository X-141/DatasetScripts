import math
from cv2 import data
import numpy as np
import cv2 as cv
import os
import re
import shutil

from TransformationMethods import FindROI, LoadImage, RefitIntoOriginalImage, CreateROIScaledImage, CreateTiltedImage, denoiseImage, thresholdBasic, thresholdOtsu

from ETL_Extraction import png_folder

png_loading_location = png_folder
target_location = "TargetSubset"
training_location = os.path.join(target_location, "training")
testing_location = os.path.join(target_location, "testing")

knn_file = os.path.join(target_location, "kNN_ETL_Subset.opknn")
dict_file = os.path.join(target_location, "kNNDictionary.txt")

# We will be loading only 10 characters identified by their
# pronunciation.
target_characters = ["A", "I", "U", "E", "O", "KA", "N", "TA", "ME", "WA"]
total_png_testing_chars = 25
total_png_training_chars = 200 - total_png_testing_chars

# Some globals to influence image transformation.
target_dimensions = 48

target_tilt = 6.0

def CreateSubsetDirectory():
    print("Loading {} training and {} test images of characters: {}".format(total_png_training_chars, \
        total_png_testing_chars, target_characters))

    if not os.path.exists(png_loading_location):
        print("Location to load pngs from does not exist. Exiting")
        exit(-1)

    if not os.path.exists(target_location):
        os.mkdir(target_location)
        os.mkdir(training_location)
        os.mkdir(testing_location)

    for character in target_characters:
        regex = re.compile(r"\d*_{}.png".format(character))
        current_train_img_total = 0
        current_test_img_total = 0
        for pngs in os.listdir(png_loading_location):
            if regex.fullmatch(pngs) is not None:
                if current_train_img_total != total_png_training_chars:
                    # print(os.path.join(os.path.abspath(png_loading_location), pngs))
                    shutil.copy(os.path.join(os.path.abspath(png_loading_location), pngs), \
                        training_location)
                    current_train_img_total = current_train_img_total + 1
                    #if current_total == total_png_per_char:
                    #    break
                elif current_test_img_total != total_png_testing_chars:
                    shutil.copy(os.path.join(os.path.abspath(png_loading_location), pngs), \
                        testing_location)
                    current_test_img_total = current_test_img_total + 1
            

    TrainSubsetWithKNN()
    CreateAffineTestData()

def TrainSubsetWithKNN():
    print("Executing training ETL model.")

    data_set = os.listdir(training_location)
    dataset_size = len(data_set)
    x_train = np.empty(shape=[dataset_size, target_dimensions, target_dimensions])
    y_train = np.empty(shape=[dataset_size, 1])
    regex = re.compile('\d*_(.*).png')

    character_dict = dict()
    character_value = 0
    row_index = 0
    
    for entry in data_set:
        filepath = os.path.join(training_location, entry)
        if os.path.exists(filepath):
            img = LoadImage(filepath)
            
            #cv.imshow("Before", img)
    
            new_img = thresholdOtsu(img)
            roi_dim = FindROI(new_img)
            roi = new_img[roi_dim[2]:roi_dim[3], roi_dim[0]:roi_dim[1]]
            centered_roi = RefitIntoOriginalImage(img, roi)
            
            #cv.imshow("After", centered_roi)
            #cv.waitKey()
            #cv.destroyAllWindows()

            target_img = cv.resize(centered_roi, (target_dimensions,target_dimensions), interpolation=cv.INTER_AREA)
            target_img = thresholdBasic(target_img)
            character = regex.match(entry).group(1)
            
            # print(character)
            #cv.imshow("Example", target_img)
            #cv.waitKey(0)
            #cv.destroyAllWindows()
            
            if not (character in character_dict):
                # print(entry)
                character_dict[character] = character_value
                character_value = character_value + 1

            os.remove(filepath)
            cv.imwrite(os.path.join(training_location, "{}_".format(row_index) + character + ".png"), centered_roi)

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
        
def CreateAffineTestData():
    data_set = os.listdir(testing_location)

    regex = re.compile("\d*_(.*).png")

    character_dict = dict()
    for x in target_characters:
        character_dict[x] = 1

    for entry in data_set:
        
        character = regex.match(entry).group(1)
        img = LoadImage(os.path.join(testing_location, entry))
        
        #denoise_img = denoiseImage(img.copy())
        #new_img = thresholdBasic(denoise_img)
        new_img = thresholdOtsu(img.copy())
        roi_dim = FindROI(new_img)
        roi = new_img[roi_dim[2]:roi_dim[3], roi_dim[0]:roi_dim[1]]
        centered_roi = RefitIntoOriginalImage(img, roi)

        roi_small = CreateROIScaledImage(img.copy(), .90)
        roi_large = CreateROIScaledImage(img.copy(), 1.15)
        roi_left_tilt = CreateTiltedImage(img.copy(), target_tilt)
        roi_right_tilt = CreateTiltedImage(img.copy(), -target_tilt)

        #cv.waitKey(0)
        #cv.destroyAllWindows()

        # these images are blacklisted due to threshold errors
        # that get past current checks.
        if character_dict[character] == 51 and character == "I":
            os.remove(os.path.join(testing_location, entry))
            character_dict[character] = character_dict[character] + 1
            continue

        os.remove(os.path.join(testing_location, entry))
        cv.imwrite(os.path.join(testing_location, "{}_".format(character_dict[character]) + character + ".png"), centered_roi)
        cv.imwrite(os.path.join(testing_location, "{}_".format(character_dict[character]+1) + character + ".png"), roi_small)
        cv.imwrite(os.path.join(testing_location, "{}_".format(character_dict[character]+2) + character + ".png"), roi_large)
        cv.imwrite(os.path.join(testing_location, "{}_".format(character_dict[character]+3) + character + ".png"), roi_left_tilt)
        cv.imwrite(os.path.join(testing_location, "{}_".format(character_dict[character]+4) + character + ".png"), roi_right_tilt)
        character_dict[character] = character_dict[character] + 5


if __name__ == "__main__":
    CreateSubsetDirectory()
