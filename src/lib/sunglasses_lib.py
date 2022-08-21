import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def uses_sunglasses(img, model, show_details=False):
    x = tf.image.resize(img, (160, 160))
    x = np.expand_dims(x, axis=0)

    predictions = model.predict_on_batch(x).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    if show_details:
        print('Predictions:\n', predictions.numpy())
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(3, 3, 1)
        plt.imshow(img)
        # plt.title(class_names[predictions[0]])
        plt.axis("off")

    return not predictions.numpy()[0]
