import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from lib import general_lib, model_lib


def sort_eyes(directory):
    path_eyes = directory + '/output/eyes'
    path_result = directory + '/output/results'
    eyes_classification = pd.read_csv(path_result + '/eyes_classification.csv', index_col=0)

    for index in range(eyes_classification.shape[0]):
        image_name = eyes_classification.file[index]
        classification = eyes_classification.classification[index]
        general_lib.move_file(image_name, path_eyes, path_eyes + '/' + classification)


def classify_eyes(path, threshold=0.0015):
    image_list = os.listdir(path+'/output/eyes')
    image_list = general_lib.filter_images(image_list)

    model_eye = model_lib.load_model('model_eye_right')

    results = pd.DataFrame()

    for image_name in image_list:
        eye_image = tf.keras.preprocessing.image.load_img(path+'/output/eyes/' + image_name)
        eye_image = tf.keras.preprocessing.image.img_to_array(eye_image)
        eye_open = open_eye_prediction(eye_image, model_eye)

        naming = image_name.split('_')
        results = pd.concat([results, pd.DataFrame({'file': image_name,
                                                    'name': naming[0],
                                                    'num': naming[1],
                                                    'side': naming[3],
                                                    'open': eye_open}, [0])])

    results['classification'] = results.open.apply(lambda x: eye_classifier(x,threshold))

    general_lib.create_folder(path+'/output/results')
    results.reset_index(drop=True).to_csv(path+'/output/results/eyes_classification.csv')


def store_eyes_single(image_name, directory):
    path = directory + '/output/faces'
    store_path = directory + '/output/eyes'
    face_image = tf.keras.preprocessing.image.load_img(path + '/' + image_name)
    face_image = tf.keras.preprocessing.image.img_to_array(face_image)
    eye_right, eye_mirror = get_eyes(face_image)

    general_lib.save_image(eye_right, image_name.split('.')[0] + '_eye_right.jpg', store_path)
    general_lib.save_image(eye_mirror, image_name.split('.')[0] + '_eye_mirror.jpg', store_path)


def store_eyes_from_directory(directory):
    image_list = os.listdir(directory)
    image_list = general_lib.filter_images(image_list)

    for image_name in image_list:
        store_eyes_single(image_name, directory)


def get_eyes(face_image, amplitude=0.45, height=0.1, wide=0.1):
    eye_right = face_image[round(face_image.shape[0] * height):round(face_image.shape[0] * (height + amplitude)),
                round(face_image.shape[1] * wide):round(face_image.shape[1] * (amplitude + wide))]

    eye_left = face_image[round(face_image.shape[0] * height):round(face_image.shape[0] * (height + amplitude)),
               round(face_image.shape[1] * (1 - (amplitude + wide))):round(face_image.shape[1] * (1 - wide))]

    eye_mirror = tf.image.flip_left_right(eye_left)
    return eye_right, eye_mirror


def eye_classifier(eye_open, threshold=0.0015):
    if eye_open > 1 - threshold:
        return 'open'
    elif eye_open < threshold:
        return 'closed'
    else:
        return 'unknown'


def eye_unknown(eye_open, threshold):
    return False if eye_open > (1 - threshold) or eye_open < threshold else True


def open_eye_prediction(img, model, show_details=False):
    x = tf.image.resize(img, (160, 160))
    x = np.expand_dims(x, axis=0)
    predictions = model.predict_on_batch(x).flatten()

    predictions = tf.nn.sigmoid(predictions)

    if show_details:
        #         print('Predictions:\n', predictions.numpy())
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(3, 3, 1)
        plt.imshow((img / 255))
        plt.title(str(predictions[0].numpy()))
        plt.axis("off")

    #     predictions = tf.where(predictions < 0.5, 0, 1)

    return predictions.numpy()[0]


def open_eyes_in_face_prediction(img_face, model_right, show=False):
    eye_right, eye_mirror = get_eyes(np.array(img_face))

    eye_right_open = open_eye_prediction(eye_right, model_right)
    eye_mirror_open = open_eye_prediction(eye_mirror, model_right)

    if show:
        print('Eye right:', eye_right_open)
        print('Eye left:', eye_mirror_open)

    return eye_right_open, eye_mirror_open
