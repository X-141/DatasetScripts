import numpy as np
import cv2 as cv
import csv
import os

csv_file_location = "data/csv/"
png_file_location = "data/png/"

csv_file_1 = "ETL7LC_01.csv"
png_files_1 = ["ETL7LC_1_00.png", "ETL7LC_1_01.png", "ETL7LC_1_02.png", "ETL7LC_1_03.png", "ETL7LC_1_04.png"]
csv_file_2 = "ETL7LC_02.csv"
png_files_2 = ["ETL7LC_2_00.png", "ETL7LC_2_01.png", "ETL7LC_2_02.png", "ETL7LC_2_03.png"]

csv_file_3 = "ETL7SC_01.csv"
png_files_3 = ["ETL7SC_1_00.png", "ETL7SC_1_01.png", "ETL7SC_1_02.png", "ETL7SC_1_03.png", "ETL7SC_1_04.png"]
csv_file_4 = "ETL7SC_02.csv"
png_files_4 = ["ETL7SC_2_00.png", "ETL7SC_2_01.png", "ETL7SC_2_02.png", "ETL7SC_2_03.png"]

character_entries = []

entry_index = 0

def extractImages(csv_file, png_files, disable_thresholding):
    character_entries.clear()
    print("opening ", csv_file)
    target_csv_file = csv_file_location + csv_file
    with open(target_csv_file, newline='') as read_file:
        reader = csv.DictReader(read_file)
        for row in reader:
            character_entries.append(row['Character Code'].strip())
    
    if not os.path.exists("SlicedData/test"):
        os.mkdir("SlicedData")
        os.mkdir("SlicedData/test")

    png_index = 1
    global entry_index
    x = 0
    y = 0

    img = cv.imread(png_file_location + png_files[0])
    
    kernel = np.ones((2,2), np.uint8)
    with open("SlicedData/gt.txt", "a") as gt_file:
        for entry in character_entries:
            new_png_name = str(entry_index)+"_"+entry+".png"
            new_png_path = "SlicedData/test/" + new_png_name
            
            sub_png_img = img[y:y+63, x:x+64]
            if not disable_thresholding:
                ret, sub_png_img = cv.threshold(sub_png_img, 150, 255, cv.THRESH_BINARY)

            sub_png_img = cv.cvtColor(sub_png_img, cv.COLOR_BGR2GRAY)
            dilation = cv.dilate(sub_png_img, kernel, iterations=1)
            
            # cv.imshow("sample", dilation)
            # cv.waitKey()
            # cv.destroyAllWindows()

            cv.imwrite(new_png_path, dilation)
            gt_file.writelines(new_png_path + "\t" + entry + "\n")
            entry_index = entry_index + 1
            x = x + 64
            if x == 3200:
                x = 0
                y = y + 63
                if y >= 2520:
                    y =  0
                    img = cv.imread(png_file_location + png_files[png_index])
                    png_index = png_index + 1


# New implementation keeps more details of the image. 
# uing normal threshold removes a lot of "smoothness"
# However, this function is incredibly slow! It is possible
# to add multi threading to this function.
def newExtractImages(csv_file, png_files):
    character_entries.clear()
    print("opening ", csv_file)
    target_csv_file = csv_file_location + csv_file
    with open(target_csv_file, newline='') as read_file:
        reader = csv.DictReader(read_file)
        for row in reader:
            character_entries.append(row['Character Code'].strip())
    
    if not os.path.exists("SlicedData/test"):
        os.mkdir("SlicedData")
        os.mkdir("SlicedData/test")

    png_index = 1
    global entry_index
    x = 0
    y = 0

    img = cv.imread(png_file_location + png_files[0])
    kernel = np.ones((2,2), np.uint8)
    with open("SlicedData/gt.txt", "a") as gt_file:
        for entry in character_entries:
            new_png_name = str(entry_index)+"_"+entry+".png"
            new_png_path = "SlicedData/test/" + new_png_name
            
            sub_png_img = img[y:y+63, x:x+64]
            sub_png_img = cv.cvtColor(sub_png_img, cv.COLOR_BGR2GRAY)
            
            denoised_img = cv.fastNlMeansDenoising(sub_png_img, None, h=40)
            adapt_thresh = cv.adaptiveThreshold(denoised_img, 255,
                                   cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 23, -2)
            dilation_adapt_thresh = cv.dilate(adapt_thresh, kernel, iterations=1)
            
            #ret, hard_thresh = cv.threshold(sub_png_img, 150, 255, cv.THRESH_BINARY)
            #dilation_hard_thresh = cv.dilate(hard_thresh, kernel, iterations=1)

            #cv.imshow("Base image", sub_png_img)
            ##cv.imshow("Denoised Image", denoised_img)
            #cv.imshow("adapt_thresh", adapt_thresh)
            #cv.imshow("dilation_adapt_thresh", dilation_adapt_thresh)
            #
            #cv.imshow("hard_thresh", hard_thresh)
            #cv.imshow("dilation_hard_thresh", dilation_hard_thresh)e
            #cv.waitKey()
            #cv.destroyAllWindows()

            cv.imwrite(new_png_path, dilation_adapt_thresh)
            gt_file.writelines(new_png_path + "\t" + entry + "\n")
            entry_index = entry_index + 1
            x = x + 64
            if x == 3200:
                x = 0
                y = y + 63
                if y >= 2520:
                    y =  0
                    img = cv.imread(png_file_location + png_files[png_index])
                    png_index = png_index + 1    

if __name__ == "__main__":

    

    newExtractImages(csv_file_1, png_files_1)
    newExtractImages(csv_file_2, png_files_2)
    newExtractImages(csv_file_3, png_files_3)
    newExtractImages(csv_file_4, png_files_4)

    disable_thresholding = False

    # extractImages(csv_file_1, png_files_1, disable_thresholding)
    # extractImages(csv_file_2, png_files_2, disable_thresholding)
    # extractImages(csv_file_3, png_files_3, disable_thresholding)
    # extractImages(csv_file_4, png_files_4, disable_thresholding)
    