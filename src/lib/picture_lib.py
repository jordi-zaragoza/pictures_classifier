import os

import pandas as pd

from lib import general_lib, image_format, blurr_lib, face_lib, eye_lib, sunglasses_lib, profile_lib, \
    manual_labeling_lib, model_lib


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


def classifier(directory='./output'):
    # directory_pictures = directory + '/pictures'
    # # Step 0: Sort blurry pictures
    # blurr_lib.sort_blurry_image(directory_pictures, blurry_rate=0.9)
    #
    # # Step 1: Retrieve-Store faces from each picture
    # face_lib.store_faces_from_directory(directory_pictures, min_proportion=0.01, min_size=80)
    #
    # # Step 2: Sort faces (valid, blurry/not-valid/sunglasses)
    # blurr_lib.sort_blurry_image(directory_pictures + '/faces', blurry_rate=0.6)
    # profile_lib.sort_profile_image(directory_pictures + '/faces', profile_sure=0.999)
    # sunglasses_lib.sort_sunglasses_faces(directory_pictures + '/faces', sunglasses_sure=0.6)
    #
    # # Step 3: Retrieve eyes
    # eye_lib.store_eyes_from_directory(directory_pictures + '/faces')
    #
    # # Step 4: Classify eyes/faces/pictures
    # eye_lib.classify_eyes(directory_pictures + '/faces/eyes')
    # face_lib.classify_faces(directory_pictures + '/faces/eyes')
    #
    # # Step 5: Sort eyes,faces,pictures (open, closed, unknown)
    # eye_lib.sort_eyes(directory_pictures + '/faces/eyes')
    # face_lib.sort_faces(directory_pictures + '/faces')
    # sort_pictures(directory_pictures + '/')
    #
    # # Step 6: move new dataset directories
    # rearrange_folders(directory)
    #
    # # Step 7: Check all datasets
    # # Step 8: Manual labeling unknown eyes folder and generate train-test
    # manual_labeling_lib.label_eyes_from_folder(directory + '/datasets_extracted/eyes/unknown')
    # general_lib.train_test_split(path='directory + /datasets_extracted/eyes')

    # Step 8: Retrain the models adding the new datasets
    model_lib.train_all_models(PATH='../data/datasets/model')


def classify_folders(path_folders):
    folders = os.listdir(path_folders)
    # Process images
    for folder in folders:
        classifier(path_folders + folder + '/pictures')


def rearrange_folders(directory):
    directory_pictures = directory + '/pictures'
    directory_datasets = directory + '/datasets_extracted'
    general_lib.move_folder(directory_pictures + '/blurry', directory_datasets + '/blurry/blurry_pic')
    general_lib.move_folder(directory_pictures + '/faces/blurry', directory_datasets + '/blurry_fac')
    general_lib.move_folder(directory_pictures + '/faces/profile', directory_datasets + '/profile')
    general_lib.move_folder(directory_pictures + '/faces/sunglasses', directory_datasets + '/sunglasses')
    general_lib.move_folder(directory_pictures + '/faces/eyes/closed', directory_datasets + '/eyes/closed')
    general_lib.move_folder(directory_pictures + '/faces/eyes/open', directory_datasets + '/eyes/open')
    general_lib.move_folder(directory_pictures + '/faces/eyes/unknown', directory_datasets + '/eyes/unknown')

    general_lib.move_folder(directory_pictures + '/faces/eyes', directory + '/eyes')
    general_lib.move_folder(directory_pictures + '/faces', directory + '/faces')
