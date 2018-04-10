"""Check image augmentation."""

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

keyword = 'bedroom'
img = load_img('data/train/%s/%s001.jpg' % (keyword, keyword))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

if not os.path.isdir('./preview'):
    os.makedirs('./preview')

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix=keyword, save_format='jpg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely