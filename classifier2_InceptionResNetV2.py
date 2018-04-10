"""
A 4 classes room classifier on top of InceptionResNetV2.

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created bedroom/, bathroom/, livingroom/ and kitchen/ subfolders inside train/ and validation/
- put the corresponding training pictures in data/train/*
- put the corresponding validation pictures in data/validation/*

In summary, this is our directory structure:
```
data/
    train/
        bedroom/
            bedroom0001.jpg
            bedroom0002.jpg
            ...
        bathroom/
            bathroom0001.jpg
            bathroom0002.jpg
            ...
        livingroom/
            livingroom0001.jpg
            livingroom0002.jpg
            ...
        kitchen/
            kitchen0001.jpg
            kitchen0002.jpg
            ...
    validation/
        bedroom/
            bedroom0001.jpg
            bedroom0002.jpg
            ...
        bathroom/
            bathroom0001.jpg
            bathroom0002.jpg
            ...
        livingroom/
            livingroom0001.jpg
            livingroom0002.jpg
            ...
        kitchen/
            kitchen0001.jpg
            kitchen0002.jpg
            ...
```
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.utils import to_categorical
from keras import applications
from keras import regularizers



result_dir = 'four_classes/InceptionResNetV2/'
bn_train_path = result_dir + 'bottleneck_features_train.npz'
bn_validation_path = result_dir + 'bottleneck_features_validation.npz'
top_model_weights_path = result_dir + 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_classes = 4
nb_train_samples = 1600 # per class
nb_validation_samples = 320 # per class
img_width, img_height = 224, 224
epochs = 50
batch_size = 32
aug_factor = 1


def save_bottlebeck_features():
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # build the InceptionResNetV2 network
    model = applications.InceptionResNetV2(include_top=False, weights='imagenet')

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        train_generator, aug_factor * nb_classes * nb_train_samples // batch_size)
    train_labels = np.tile(np.repeat(np.arange(nb_classes), nb_train_samples), aug_factor)
    np.savez(bn_train_path, data=bottleneck_features_train, label=train_labels)

    test_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        test_generator, nb_classes * nb_validation_samples // batch_size)
    validation_labels = np.repeat(np.arange(nb_classes), nb_validation_samples)
    np.savez(bn_validation_path, data=bottleneck_features_validation, label=validation_labels)


def train_top_model():
    train = np.load(bn_train_path)
    train_data, train_labels = train['data'], train['label']
    # convert labels to categorical one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=nb_classes)
    # print train_data.shape, train_labels.shape

    validation = np.load(bn_validation_path)
    validation_data, validation_labels = validation['data'], validation['label']
    # convert labels to categorical one-hot encoding
    validation_labels = to_categorical(validation_labels, num_classes=nb_classes)
    # print validation_data.shape, validation_labels.shape

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0012)))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

    # print(history.history.keys())

    # visulization
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(result_dir + 'accuracy.png')
    plt.close()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(result_dir + 'loss.png')
    plt.close()


if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

if not (os.path.exists(bn_train_path) and os.path.exists(bn_validation_path)):
    save_bottlebeck_features()

train_top_model()
