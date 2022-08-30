import os

import pandas as pd

from lib import general_lib, image_format, blurr_lib, face_lib, eye_lib, sunglasses_lib, profile_lib


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
        retrieve_files(path_folders+folder, path_save+folder)


def retrieve_files(directory_pictures, directory_store='./output'):
    # Convert raw images into jpg images
    image_format.images_to_jpg_folder(directory_pictures, directory_store, base_width=2042.0)


def classifier(directory_pictures='./output/pictures'):
    # Step 0: Sort blurry pictures
    blurr_lib.sort_blurry_image(directory_pictures, sharp_rate=0.9)

    # Step 1: Retrieve-Store faces from each picture
    face_lib.store_faces_from_directory(directory_pictures, min_proportion=0.01, min_size=80)

    # Step 2: Sort faces (valid, blurry/not-valid/sunglasses)
    blurr_lib.sort_blurry_image(directory_pictures+'/faces', sharp_rate=.9)
    profile_lib.sort_profile_image(directory_pictures+'/faces', profile_rate=.99)
    sunglasses_lib.sort_sunglasses_faces(directory_pictures+'/faces', sunglasses_rate=0.9)

    # # Step 3: Retrieve eyes
    # eye_lib.store_eyes_from_directory(directory_pictures + '/faces')
    #
    # # Step 4: Classify eyes/faces/pictures
    # eye_lib.classify_eyes(directory_pictures + '/faces/eyes')
    # face_lib.classify_faces(directory_pictures + '/faces/eyes')

    # # Step 5: Sort eyes,faces,pictures (open, closed, unknown)
    # eye_lib.sort_eyes(directory_pictures + '/faces/eyes')
    # face_lib.sort_faces(directory_pictures + '/faces')
    # sort_pictures(directory_pictures + '/')


def classify_folders(path_folders):
    folders = os.listdir(path_folders)
    # Process images
    for folder in folders:
        classifier(path_folders+folder+'/pictures')
