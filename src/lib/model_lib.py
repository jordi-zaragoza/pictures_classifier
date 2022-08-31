import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from lib import general_lib


def train_all_models(PATH='../data/datasets/model', BATCH_SIZE=32, IMG_SIZE=(160, 160)):
    model_trainer(PATH + '/eye_use', 'model_eye', BATCH_SIZE, IMG_SIZE)
    model_trainer(PATH + '/blurry_use', 'model_blurry', BATCH_SIZE, IMG_SIZE)
    model_trainer(PATH + '/profile_use', 'model_profile', BATCH_SIZE, IMG_SIZE)
    model_trainer(PATH + '/sunglasses_use', 'model_sunglasses', BATCH_SIZE, IMG_SIZE)


def predict(img, model, show_details=False, true_percent=0.5, false_percent=0.5):
    x = tf.image.resize(img, (160, 160))
    x = np.expand_dims(x, axis=0)

    predictions = model.predict_on_batch(x).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)

    if show_details:
        print('Predictions:\n', predictions.numpy())
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(3, 3, 1)
        plt.imshow(img.astype(np.uint8))
        plt.title([predictions[0]])
        # plt.axis("off")

    is_true = tf.where(predictions > true_percent, 1, 0)
    is_false = tf.where(predictions < (1-false_percent), 1, 0)

    # print(predictions.numpy()[0])
    # print(is_false.numpy()[0])

    return is_true.numpy()[0], is_false.numpy()[0]


def save_model(model, model_name):
    general_lib.create_folder('model')
    model_json = model.to_json()
    with open('model/' + model_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('model/' + model_name + "_weights.h5")
    print("Saved model to disk")


def load_model(model_name):
    # load json and create model
    json_file = open('model/' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)

    # load weights
    loaded_model.load_weights('model/' + model_name + '_weights.h5')

    return loaded_model


def get_train_val_test_sets(PATH, BATCH_SIZE, IMG_SIZE, show=False):
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'val')
    test_dir = os.path.join(PATH, 'test')

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                     shuffle=True,
                                                                     batch_size=BATCH_SIZE,
                                                                     image_size=IMG_SIZE)
    if show:
        plt.figure(figsize=(10, 10))
        for images, labels in train_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(train_dataset.class_names[labels[i]])
                plt.axis("off")

    if os.path.exists(test_dir):
        test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                   shuffle=True,
                                                                   batch_size=BATCH_SIZE,
                                                                   image_size=IMG_SIZE)

    else:
        val_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(val_batches // 5)
        validation_dataset = validation_dataset.skip(val_batches // 5)

        print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
        print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    return train_dataset, validation_dataset, test_dataset


def autotune_sets(train_dataset, validation_dataset, test_dataset):
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, test_dataset


def data_augmentation_layer(train_dataset):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.5),
        tf.keras.layers.RandomZoom(0.05),
    ])

    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')

    return data_augmentation


# Create the base model from the pre-trained model MobileNet V2
def create_base_model(IMG_SIZE):
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    base_model.summary()
    return base_model


def create_model(train_dataset, base_model, data_augmentation):
    image_batch, _ = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def model_evaluate(model, test_dataset, class_names):
    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")


def add_metrics(history, acc=0, val_acc=0, loss=0, val_loss=0):
    acc += history.history['accuracy']
    val_acc += history.history['val_accuracy']
    loss += history.history['loss']
    val_loss += history.history['val_loss']
    return acc, val_acc, loss, val_loss


def get_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return acc, val_acc, loss, val_loss


def plot_metrics_history(acc, val_acc, loss, val_loss, initial_epochs=0):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    # plt.ylim([0.8, 1])
    plt.title('Training and Validation Accuracy')

    if initial_epochs != 0:
        plt.plot([initial_epochs - 1, initial_epochs - 1],
                 plt.ylim(), label='Start Fine Tuning')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])

    if initial_epochs != 0:
        plt.plot([initial_epochs - 1, initial_epochs - 1],
                 plt.ylim(), label='Start Fine Tuning')

    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def model_trainer(PATH, MODEL_NAME, BATCH_SIZE, IMG_SIZE):
    train_dataset_i, validation_dataset_i, test_dataset_i = get_train_val_test_sets(PATH, BATCH_SIZE, IMG_SIZE)
    class_names = train_dataset_i.class_names
    data_augmentation = data_augmentation_layer(train_dataset_i)

    train_dataset, validation_dataset, test_dataset = autotune_sets(train_dataset_i, validation_dataset_i,
                                                                    test_dataset_i)

    # Base Model -------------------------------------------------------------------
    base_model = create_base_model(IMG_SIZE)
    model = create_model(train_dataset, base_model, data_augmentation)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    len(model.trainable_variables)

    initial_epochs = 10

    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

    acc, val_acc, loss, val_loss = get_metrics(history)
    plot_metrics_history(acc, val_acc, loss, val_loss)

    # Fine tuning --------------------------------------------------------------
    base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10),
                  metrics=['accuracy'])

    model.summary()

    len(model.trainable_variables)

    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)

    acc, val_acc, loss, val_loss = add_metrics(history_fine, acc, val_acc, loss, val_loss)
    plot_metrics_history(acc, val_acc, loss, val_loss, initial_epochs)

    # Verify prediction ----------------------------------------------------------------------
    model_evaluate(model, test_dataset, class_names)

    # Save model ---------------------------------------------------------------------------------
    save_model(model, model_name=MODEL_NAME)
