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

# Loading the Colormap
colormap = loadmat(
    "human_colormap.mat"
)["colormap"]
colormap = colormap * 100
print(colormap)
colormap = colormap.astype(np.uint8)

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 4

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

# Khoi tao
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)


# Load model de test
model.load_weights("checkpoint.hdf5")

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

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def Unet_detect_position(link_image):
    contour_points = []

    image_tensor = read_image(link_image)
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 4)


    ###### RGB
    unique_values = colormap[1:NUM_CLASSES]  # Loại bỏ lớp nền
    key_points = []
    contour_points = []

    for idx, value in enumerate(unique_values):
        # print(f"Processing color: {value}")
        # Tạo mặt nạ nhị phân để giữ lại lớp có màu chính xác bằng value
        mask = cv2.inRange(prediction_colormap, np.array(value), np.array(value))

        # Tìm các contour trong mặt nạ
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour_part in range(len(contours)):
            area=cv2.contourArea(contours[contour_part])
            if area>120 and area<1000000:
                # cv2.drawContours(image,contours[contour_part],-1,(255,0,0),2)
                esp= 0.02*cv2.arcLength(contours[contour_part],True)
                approx=(cv2.approxPolyDP(contours[contour_part],esp, True))
                #### phan này anh danh theo stt của lớp anh đặt em không rõ nên đặt bừa 
                if idx == 0:
                    key= 'Class1'
                elif idx== 1:
                    key= 'Class2'
                else:
                    key= "NG"
                key_points.append(key)
                contour_points.append(approx)
    return contour_points,key_points