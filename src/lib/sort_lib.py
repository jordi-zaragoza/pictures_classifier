import os
import tensorflow as tf

from lib import general_lib, model_lib


def sort_blurry_image(path, blurry_sure=.95, show_details=False):
    image_list = os.listdir(path)
    image_list = general_lib.filter_images(image_list)

    model_blurry = model_lib.load_model('model_blurry')

    for image_name in image_list:
        image = tf.keras.preprocessing.image.load_img(path + '/' + image_name)
        image = tf.keras.preprocessing.image.img_to_array(image)

        sharp, blurry = model_lib.predict(image, model_blurry, show_details, false_percent=blurry_sure)

        if blurry:
            general_lib.move_file(image_name, path, path + '/blurry')


def sort_sunglasses_faces(path_faces, sunglasses_sure=.9, show_details=False):
    image_list = os.listdir(path_faces)
    image_list = general_lib.filter_images(image_list)

    model_sunglasses = model_lib.load_model('model_sunglasses')

    for image_name in image_list:
        face_image = tf.keras.preprocessing.image.load_img(path_faces + '/' + image_name)
        face_image = tf.keras.preprocessing.image.img_to_array(face_image)

        _, sunglasses = model_lib.predict(face_image, model_sunglasses,
                                          show_details=show_details, false_percent=sunglasses_sure)
        # print(sunglasses)
        if sunglasses == 1:
            general_lib.move_file(image_name, path_faces, path_faces + '/sunglasses')


def sort_profile_image(path, profile_sure=.99, show_details=False):
    image_list = os.listdir(path)
    image_list = general_lib.filter_images(image_list)

    model_blurry = model_lib.load_model('model_profile')

    for image_name in image_list:
        image = tf.keras.preprocessing.image.load_img(path + '/' + image_name)
        image = tf.keras.preprocessing.image.img_to_array(image)

        profile, _ = model_lib.predict(image, model_blurry, show_details=show_details, true_percent=profile_sure)

        if profile:
            general_lib.move_file(image_name, path, path + '/profile')