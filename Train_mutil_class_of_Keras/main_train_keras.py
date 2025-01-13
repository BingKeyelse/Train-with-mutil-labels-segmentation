import keras
from keras import layers
from tensorflow.python.framework import ops


import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

# For data preprocessing
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

from layer import DeeplabV3Plus

#######################
####
IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 4
DATA_DIR = "Data"
# NUM_TRAIN_IMAGES = 1000
# NUM_VAL_IMAGES = 50

train_images = sorted(glob(os.path.join(DATA_DIR, "picture/*")))[:200]
train_masks = sorted(glob(os.path.join(DATA_DIR, "mask/*")))[:200]
val_images = sorted(glob(os.path.join(DATA_DIR, "picture/*")))[
    200 : 
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "mask/*")))[
    200 : 
]


def read_image(image_path, mask=False):
    image = tf_io.read_file(image_path)
    if mask:
        image = tf_image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
# model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

from keras.callbacks import ModelCheckpoint

filepath= "checkpoint_ver2.hdf5"
callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

history = model.fit(train_dataset, validation_data=val_dataset, epochs=35, callbacks=[callback])


