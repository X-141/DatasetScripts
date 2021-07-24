import cv2 as cv
import csv
import os


### IMPORTANT:
# You need the ETL dataset from:
# http://etlcdb.db.aist.go.jp/
# and use the provided scripts from that site to extract the data.
#
# Then you must create the folder structure according to the two
# variables below:
data_file_location = "data"
csv_file_location = os.path.join(data_file_location, "csv")
#csv_file_location = "data/csv/"
png_file_location = os.path.join(data_file_location, "png")
#png_file_location = "data/png/"

csv_file_1 = "ETL7LC_01.csv"
png_files_1 = ["ETL7LC_1_00.png", "ETL7LC_1_01.png", "ETL7LC_1_02.png", "ETL7LC_1_03.png", "ETL7LC_1_04.png"]
csv_file_2 = "ETL7LC_02.csv"
png_files_2 = ["ETL7LC_2_00.png", "ETL7LC_2_01.png", "ETL7LC_2_02.png", "ETL7LC_2_03.png"]

csv_file_3 = "ETL7SC_01.csv"
png_files_3 = ["ETL7SC_1_00.png", "ETL7SC_1_01.png", "ETL7SC_1_02.png", "ETL7SC_1_03.png", "ETL7SC_1_04.png"]
csv_file_4 = "ETL7SC_02.csv"
png_files_4 = ["ETL7SC_2_00.png", "ETL7SC_2_01.png", "ETL7SC_2_02.png", "ETL7SC_2_03.png"]

# Store character entries from the CSV value.
# cleared each execution of extractETLImages(csv_file, png_files)
character_entries = []

# Current number of extracted images
extracted_image_index = 0

# Output directory of extracted "sliced" data
output_directory = "SlicedData"
png_folder = os.path.join(output_directory, "png")
gt_file = os.path.join(output_directory, "gt.txt")

def extractETLImages(csv_file, png_files):
    """ extractETLImages(csv_file, png_files):

        Given a csv_file containing label information, and given a list of png_files,
        extract induvidual png images from the ETL dataset.

        Each image will be assigned a numeric value according to the current number
        of extracted images and their character label.

        No processing will be applied to the extracted image.
    """
    character_entries.clear()
    print("opening ", csv_file)
    target_csv_file = os.path.join(csv_file_location, csv_file)
    # target_csv_file = csv_file_location + csv_file
    # Load character_entries with character labels from
    # csv file.
    with open(target_csv_file, newline='') as read_file:
        reader = csv.DictReader(read_file)
        for row in reader:
            character_entries.append(row['Character Code'].strip())
    
    if not os.path.exists(png_folder):
        os.mkdir(output_directory)
        os.mkdir(png_folder)

    # Used to access the indeces of passed png_files.
    png_index = 1
    global extracted_image_index
    # Used to track position in the pngs provided by dataset.
    # Each image is a 64x63, and we will access and slice images
    # row by row.
    x = 0
    y = 0

    # load in first dataset png.
    img = cv.imread(os.path.join(png_file_location, png_files[0]))
    #img = cv.imread(png_file_location + png_files[0])
    
    with open(gt_file, "a") as file:
        # Access each character entry from character_entries
        for entry in character_entries:
            new_png_name = str(extracted_image_index)+"_"+entry+".png"
            new_png_path = os.path.join(png_folder, new_png_name)
            
            sub_png_img = img[y:y+63, x:x+64]

            cv.imwrite(new_png_path, sub_png_img)
            
            file.writelines(new_png_path + "\t" + entry + "\n")
            extracted_image_index = extracted_image_index + 1
            
            # move to next image in the dataset png.
            x = x + 64

            if x == 3200: # We need to jump to next row
                x = 0
                y = y + 63
                if y >= 2520: # We have reached the end of the dataset png
                    y =  0
                    # load next dataset png.
                    #img = cv.imread(png_file_location + png_files[png_index])
                    img = cv.imread(os.path.join(png_file_location, png_files[png_index]))
                    png_index = png_index + 1    

if __name__ == "__main__":
    extractETLImages(csv_file_1, png_files_1)
    extractETLImages(csv_file_2, png_files_2)
    extractETLImages(csv_file_3, png_files_3)
    extractETLImages(csv_file_4, png_files_4)

    