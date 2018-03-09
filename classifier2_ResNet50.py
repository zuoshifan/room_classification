'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.utils import to_categorical
from keras import applications
from keras import regularizers



result_dir = 'four_classes/adam_ResNet50/'
bn_train_path = result_dir + 'bottleneck_features_train.npz'
bn_validation_path = result_dir + 'bottleneck_features_validation.npz'
top_model_weights_path = result_dir + 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_classes = 4
nb_train_samples = 1600 # per class
nb_validation_samples = 320 # per class
img_width, img_height = 150, 150
epochs = 50
batch_size = 32
aug_factor = 1


def save_bottlebeck_features():
#     train_datagen = ImageDataGenerator(
# 	    rescale=1. / 255,
#             rotation_range=0.2,
# 	    width_shift_range=0.2,
# 	    height_shift_range=0.2,
# 	    shear_range=0.2,
# 	    zoom_range=0.2,
# 	    horizontal_flip=True)

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    # model = applications.VGG16(include_top=False, weights='imagenet')

    # build the ResNet50 network, width and height should be no smaller than 197
    model = applications.ResNet50(include_top=False, weights='imagenet')

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
