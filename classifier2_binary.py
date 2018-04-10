"""
A first trier of a binary room classifier.

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created bedroom/ and bathroom/ subfolders inside train/ and validation/
- put the bedroom training pictures in data/train/bedroom
- put the bedroom validation pictures in data/validation/bedroom
- put the bathroom training  pictures in data/train/bathroom
- put the bathroom validation pictures in data/validation/bathroom

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
    validation/
        bedroom/
            bedroom0001.jpg
            bedroom0002.jpg
            ...
        bathroom/
            bathroom0001.jpg
            bathroom0002.jpg
            ...
```
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import regularizers

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 3200
nb_validation_samples = 704
epochs = 100
batch_size = 32


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

    print(history.history.keys())

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
    plt.savefig('accuracy.png')
    plt.close()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('loss.png')
    plt.close()


# save_bottlebeck_features()
train_top_model()
