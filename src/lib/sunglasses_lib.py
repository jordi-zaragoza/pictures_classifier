import os
import tensorflow as tf

from lib import general_lib, model_lib


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
