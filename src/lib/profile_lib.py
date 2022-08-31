import os
import tensorflow as tf
from lib import general_lib, model_lib


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
