from lib import model_lib
import tensorflow as tf
import os

PATH = '../datasets/eye_use'
MODEL_NAME = 'model_eye'

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# Model evaluation
test_dir = os.path.join(PATH, 'test')
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           shuffle=True,
                                                           batch_size=BATCH_SIZE,
                                                           image_size=IMG_SIZE)

base_learning_rate = 0.0001
model = model_lib.load_model(model_name='model_eye_right')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_lib.model_evaluate(model, test_dataset, test_dataset.class_names)