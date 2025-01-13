# from sklearn.model_selection import train_test_split
import os
import random

import tensorflow as tf
import cv2
import numpy as np

import os
os.environ["SM_FRAMEWORK"]="tf.keras"
import matplotlib.pyplot as plt


# Import thu vien segmentation_models
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
sm.set_framework("tf.keras")
sm.framework()

BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)


# Khoi tao model
opt=tf.keras.optimizers.Adam(0.001)
model= Unet(BACKBONE,encoder_weights="imagenet",classes=1,activation="sigmoid",input_shape=(512,512,3),encoder_freeze=True)
loss1 = sm.losses.categorical_focal_dice_loss
model.compile(optimizer=opt,loss=loss1,metrics=[iou_score])


# Load model de test
model.load_weights("checkpoint_of_PVN_new.hdf5")


def Unet_detect_position(link_image):
    contour_points = []

    # for id in index:

    # Anh dau vao, ko phai mask
    image = cv2.imread(link_image,1)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    # Dua qua model de predicted segmentation map

    mask_predict = model.predict(image[np.newaxis, :, :, :])

    z = mask_predict[0]#[:, :, 0]
    im = np.asarray(z)
    im1 = im*255
    cv2.imwrite(r"D:\ALL_of_data_PVN_to_use_tool\picture.png", im1)

    image_process= cv2.imread(r"D:\ALL_of_data_PVN_to_use_tool\picture.png", 0)
    ret, thread= cv2.threshold(image_process,50,255,cv2.THRESH_BINARY)
    contour,_= cv2.findContours(thread,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour_part in range(len(contour)):
        area=cv2.contourArea(contour[contour_part])
        if area>60 and area<1000000:
            cv2.drawContours(image,contour[contour_part],-1,(255,0,0),2)
            esp= 0.02*cv2.arcLength(contour[contour_part],True)
            approx=(cv2.approxPolyDP(contour[contour_part],esp, True))
            contour_points.append(approx)
    return contour_points