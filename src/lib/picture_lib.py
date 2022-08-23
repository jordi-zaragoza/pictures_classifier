import pandas as pd

from lib import general_lib, image_format, blurr_lib, face_lib, eye_lib


def sort_pictures(directory):
    path_eyes = directory
    path_result = directory + 'faces/eyes/results'
    eyes_classification = pd.read_csv(path_result + '/pictures_classification.csv', index_col=0)

    for index in range(eyes_classification.shape[0]):
        image_name = eyes_classification.name[index] + '.jpg'
        closed_faces = eyes_classification.closed[index]
        if closed_faces > 0:
            general_lib.move_file(image_name, path_eyes, path_eyes + '/closed_eyes')


def classifier(directory):
    # Convert raw images into jpg images
    image_format.images_to_jpg_folder(directory, base_width=2042.0)

    # Step 0: Sort blurry pictures
    blurr_lib.blurr_classifier_folder(directory + '/output/pictures', grid_size=(3, 3), cpbd=False, motion=False)
    blurr_lib.blurr_sort(directory + '/output/pictures', threshold=(100, 0, 0))

    # Step 1: Retrieve-Store faces from each picture
    face_lib.store_faces_from_directory(directory + '/output/pictures', min_proportion=0.01, min_size=80)

    # Step 2: Sort faces (valid, blurry/not-valid/sunglasses)
    blurr_lib.blurr_classifier_folder(directory + '/output/pictures/faces', file_name='faces_blur.csv',
                                      grid_size=[1, 1])
    blurr_lib.blurr_sort(directory + '/output/pictures/faces', file_name='faces_blur.csv', threshold=[4, 20, 0])
    face_lib.sort_sunglasses(directory + '/output/pictures/faces')

    # Step 3: Retrieve eyes
    eye_lib.store_eyes_from_directory(directory + '/output/pictures/faces')

    # Step 4: Classify eyes/faces/pictures
    eye_lib.classify_eyes(directory + '/output/pictures/faces/eyes')
    face_lib.classify_faces(directory + '/output/pictures/faces/eyes')

    # Step 5: Sort eyes,faces,pictures (open, closed, unknown)
    eye_lib.sort_eyes(directory + '/output/pictures/faces/eyes')
    face_lib.sort_faces(directory + '/output/pictures/faces')
    sort_pictures(directory + '/output/pictures/')
