import os
import face_recognition
import numpy as np
import pandas as pd
import tensorflow as tf

from lib import eye_lib, model_lib, general_lib, sunglasses_lib, image_format


def sort_faces(directory):
    path_eyes = directory
    path_result = directory + '/eyes/results'
    eyes_classification = pd.read_csv(path_result + '/faces_classification.csv', index_col=0)

    for index in range(eyes_classification.shape[0]):
        image_name = eyes_classification.name[index]+'_'+str(eyes_classification.face[index])+'.jpg'
        classification = eyes_classification.classification[index]
        general_lib.move_file(image_name, path_eyes, path_eyes + '/' + classification)


def classify_faces(directory):
    path_results = directory+'/results'
    eyes_classification = pd.read_csv(path_results+'/eyes_classification.csv', index_col=0)

    faces_table = pd.DataFrame()
    picture_name_list = eyes_classification.name.unique()
    for picture_name in picture_name_list:
        df_picture = eyes_classification[eyes_classification.name == picture_name]
        face_name_list = df_picture.num.unique()
        for face_name in face_name_list:
            df_faces = df_picture[df_picture.num == face_name]
            classification = face_classifier(df_faces.classification.values)
            faces_table = pd.concat([faces_table, pd.DataFrame({'name':picture_name, 'face':face_name, 'classification': classification},[0])])

    faces_table.reset_index(inplace=True, drop=True)
    faces_table['values'] = 1
    pictures_table = faces_table.pivot_table(index='name', columns='classification', values='values', aggfunc=np.sum)
    pictures_table = pictures_table.fillna('0')
    pictures_table.reset_index(inplace=True)
    pictures_table.columns = pictures_table.columns.values

    for col in ['open', 'closed', 'unknown']:
        try:
            pictures_table[col] = pictures_table[col].astype(int)
        except:
            pictures_table[col] = 0

    general_lib.create_folder(path_results)
    faces_table.to_csv(path_results+'/faces_classification.csv')
    pictures_table.to_csv(path_results+'/pictures_classification.csv')


def store_faces_single(directory, image_name, directory_store, min_proportion=0.05, min_size=50):
    """
    This method stores the faces of a given picture on a folder called faces
    """
    image_path = directory + '/' + image_name
    base_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(base_image)

    print(image_name, "I found {} face(s) in this photograph.".format(len(face_locations)))

    for num, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = base_image[top:bottom, left:right]

        if valid_proportion(face_image, base_image, min_proportion, min_size):
            face_image_resize = image_format.resize_image(face_image, base_width=300)
            general_lib.save_image(face_image_resize, image_name.split('.')[0] + '_' + str(num), directory_store)


def store_faces_from_directory(directory, min_proportion=0.01, min_size=50):
    image_list = os.listdir(directory)
    image_list = general_lib.filter_images(image_list)

    for image_name in image_list:
        store_faces_single(directory, image_name, directory+'/faces', min_proportion, min_size)


def valid_proportion(face_image, image, min_proportion, min_size):
    valid_size = face_image.shape[0] > min_size and face_image.shape[1] > min_size
    valid_height = face_image.shape[0] > min_proportion * image.shape[0]
    valid_width = face_image.shape[1] > min_proportion * image.shape[1]
    if not valid_size:
        print('Not valid size ', face_image.shape)
    if not valid_height or not valid_width:
        print('Not valid proportion ', face_image.shape[0], '->', image.shape[0])
    return valid_height and valid_width and valid_size


def classify_face(eye_right_open, eye_mirror_open, sunglasses, threshold):
    face_unknown = eye_lib.eye_unknown(eye_right_open, threshold) or eye_lib.eye_unknown(eye_mirror_open, threshold)

    if sunglasses:
        return 'sunglasses'

    elif face_unknown:
        return 'unknown'

    else:
        return {
            0: "closed",
            1: "half",
            2: "open",
        }.get(round(eye_right_open + eye_mirror_open), "error")


def face_classifier(eye_values):
    eye1 = eye_values[0]
    eye2 = eye_values[1]
    if eye1 == 'open' and eye2 == 'open':
        return 'open'
    elif eye1 == 'closed' and eye2 == 'closed':
        return 'closed'
    else:
        return 'unknown'


def face_proportion_greater(face_image, image_base, proportion=0.1):
    return face_image.size[0] > (proportion * image_base.size[0]) and face_image.size[1] > (
            proportion * image_base.size[1])
