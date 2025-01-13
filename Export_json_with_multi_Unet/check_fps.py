from sklearn.model_selection import train_test_split
import os
import random
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from segmentation_models.metrics import*
import time

# Import thu vien segmentation_models
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
sm.set_framework("tf.keras")
sm.framework()
from keras.utils import to_categorical



BACKBONE = "resnet50"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Khoi tao
opt=tf.keras.optimizers.Adam(0.0001)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model= Unet(BACKBONE,encoder_weights='imagenet',classes=4,activation="softmax",encoder_freeze=True)
model.compile(opt, total_loss, metrics=metrics)


# Load model de test
model.load_weights("weight_data_catdog.hdf5")

def Unet_detect_position(link_image):
    
    contour_points = []

    # for id in index:

    # Anh dau vao, ko phai mask
    image = cv2.imread(link_image,1)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    image_use= np.expand_dims(image, 0)

    image_use= preprocess_input(image_use)

    mask_predict = model.predict_on_batch(image_use)
    image_prediction = np.argmax(mask_predict, axis=3)[0,:,:]

frame_count = 0
start_time = time.time()

while True:
    # Tăng bộ đếm khung hình
        frame_count += 1
        
        # Lấy thời gian hiện tại
        current_time = time.time()
        link_image='picture/data316.png'
        Unet_detect_position(link_image)
        
        # Tính khoảng thời gian đã trôi qua
        elapsed_time = current_time - start_time
        
        if elapsed_time > 1:  # Mỗi giây
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            
            # Reset bộ đếm và thời gian bắt đầu
            frame_count = 0
            start_time = current_time