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

def extractImages(csv_file, png_files):
    character_entries.clear()
    print("opening ", csv_file)
    target_csv_file = csv_file_location + csv_file
    with open(target_csv_file, newline='') as read_file:
        reader = csv.DictReader(read_file)
        for row in reader:
            character_entries.append(row['Character Code'].strip())
    
    if not os.path.exists("NEWDATA/test"):
        os.mkdir("NEWDATA")
        os.mkdir("NEWDATA/test")

    png_index = 1
    global entry_index
    x = 0
    y = 0

    img = cv.imread(png_file_location + png_files[0])
    
    kernel = np.ones((2,2), np.uint8)
    with open("NEWDATA/gt.txt", "a") as gt_file:
        for entry in character_entries:
            new_png_name = str(entry_index)+"_"+entry+".png"
            new_png_path = "NEWDATA/test/" + new_png_name
            # thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            ret, thresh = cv.threshold(img[y:y+63, x:x+64], 150, 255, cv.THRESH_BINARY)
            # ret, thresh = cv.threshold(img[y:y+63, x:x+64], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
            dilation = cv.dilate(thresh, kernel, iterations=1)
            # rect = cv.boundingRect(dilation)

            # print(rect)
            # cv.rectangle(dilation, rect, (255,255,255),0)
            
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


if __name__ == "__main__":

    extractImages(csv_file_1, png_files_1)
    extractImages(csv_file_2, png_files_2)
    extractImages(csv_file_3, png_files_3)
    extractImages(csv_file_4, png_files_4)
    