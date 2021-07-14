from keras import datasets
import numpy as np
import cv2 as cv
import os
import re

target_png_folder = "NEWDATA/test/"

def trainKnnModel_ETL():
    print("Executing training ETL model.")

    data_set = os.listdir(target_png_folder)
    dataset_size = len(data_set)
    x_train = np.empty(shape=[dataset_size, 32, 32])
    y_train = np.empty(shape=[dataset_size, 1])
    regex = re.compile('\d*_(.*).png')

    character_dict = dict()
    character_value = 0
    row_index = 0
    # kernel = np.ones((2,2), np.uint8)
    
    for entry in data_set:
        filepath = os.path.join(target_png_folder, entry)
        if os.path.exists(filepath):
            img = cv.imread(filepath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            resized = cv.resize(img, (32,32), interpolation=cv.INTER_AREA)
            ret, thresh = cv.threshold(resized, 80, 255, cv.THRESH_BINARY)
            # dilation = cv.dilate(thresh, kernel, iterations=1)
            # cv.imshow("Test", thresh)
            # cv.waitKey()
            # cv.destroyAllWindows()
            # continue
            character = regex.match(entry).group(1)
            
            if not (character in character_dict):
                # print(entry)
                character_dict[character] = character_value
                character_value = character_value + 1

            y_train[row_index] = character_dict[character]
            x_train[row_index] = thresh
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
    knn.save('knn_etl_data.opknn')

# print(entry)
# print(regex.match(entry).group(1))

# print(img.shape)

if __name__ == "__main__":
    print("Hello world")
    trainKnnModel_ETL()

    # (60000, 28, 28)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(x_train.shape)
    # X = x_train.reshape(len(x_train), -1)
    # print(X.shape)

