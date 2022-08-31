import os
import pandas as pd
from lib import general_lib, image_format


def sort_pictures(directory):
    path_eyes = directory
    path_result = directory + 'faces/eyes/results'
    eyes_classification = pd.read_csv(path_result + '/pictures_classification.csv', index_col=0)

    for index in range(eyes_classification.shape[0]):
        image_name = eyes_classification.name[index] + '.jpg'
        closed_faces = eyes_classification.closed[index]
        if closed_faces > 0:
            general_lib.move_file(image_name, path_eyes, path_eyes + '/closed_eyes')


def get_pictures_from_folders(path_folders, path_save):
    folders = os.listdir(path_folders)
    print(folders)
    # Get images from folder
    for folder in folders:
        retrieve_files(path_folders + folder, path_save + folder)


def retrieve_files(directory_pictures, directory_store='./output'):
    # Convert raw images into jpg images
    image_format.images_to_jpg_folder(directory_pictures, directory_store, base_width=2042.0)
