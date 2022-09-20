import os

from lib import face_lib, eye_lib, picture_lib, general_lib, sort_lib


def classifier(directory='./output'):

    directory_pictures = directory + '/pictures'
    # Step 0: Sort blurry pictures
    sort_lib.sort_blurry_image(directory_pictures, blurry_sure=0.999)

    # Step 1: Retrieve-Store faces from each picture
    face_lib.store_faces_from_directory(directory_pictures, min_proportion=0.01, min_size=80)

    # Step 2: Sort faces (valid, blurry/not-valid/sunglasses)
    sort_lib.sort_blurry_image(directory_pictures + '/faces', blurry_sure=0.6)
    sort_lib.sort_profile_image(directory_pictures + '/faces', profile_sure=0.999)
    sort_lib.sort_sunglasses_faces(directory_pictures + '/faces', sunglasses_sure=0.6)

    # Step 3: Retrieve eyes
    eye_lib.store_eyes_from_directory(directory_pictures + '/faces')

    # Step 4: Classify eyes/faces/pictures
    eye_lib.classify_eyes(directory_pictures + '/faces/eyes', threshold_closed=0.1, threshold_open=0.4)
    face_lib.classify_faces(directory_pictures + '/faces/eyes')

    # Step 5: Sort eyes,faces,pictures (open, closed, unknown)
    eye_lib.sort_eyes(directory_pictures + '/faces/eyes')
    face_lib.sort_faces(directory_pictures + '/faces')
    picture_lib.sort_pictures(directory_pictures + '/')

    # Step 6: move new dataset directories
    rearrange_folders(directory)


def classify_folders(path_folders):
    folders = os.listdir(path_folders)
    # Process images
    for folder in folders:
        print('================================================================')
        print('Folder to classify:', path_folders+'/'+folder)
        try:
            classifier(path_folders+'/'+folder)
        except:
            print('No pictures found on:', path_folders+'/'+folder)


def rearrange_folders(directory):
    directory_pictures = directory + '/pictures'
    directory_datasets = directory + '/datasets_extracted'
    general_lib.move_folder(directory_pictures + '/blurry', directory_datasets + '/blurry_pic')
    general_lib.move_folder(directory_pictures + '/faces/blurry', directory_datasets + '/blurry_fac')
    general_lib.move_folder(directory_pictures + '/faces/profile', directory_datasets)
    general_lib.move_folder(directory_pictures + '/faces/sunglasses', directory_datasets)
    general_lib.move_folder(directory_pictures + '/faces/eyes/closed', directory_datasets + '/eyes')
    general_lib.move_folder(directory_pictures + '/faces/eyes/open', directory_datasets + '/eyes')
    general_lib.move_folder(directory_pictures + '/faces/eyes/unknown', directory_datasets + '/eyes')
    general_lib.move_folder(directory_pictures + '/faces/eyes/results', directory)