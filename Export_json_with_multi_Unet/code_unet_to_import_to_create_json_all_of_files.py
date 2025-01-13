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
model.load_weights("weight.hdf5")


def Unet_detect_position(link_image):
    contour_points = []

    # for id in index:

    # Anh dau vao, ko phai mask
    image = cv2.imread(link_image,1)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    image_use= np.expand_dims(image, 0)

    image_use= preprocess_input(image_use)

    mask_predict = model.predict(image_use)
    image_prediction = np.argmax(mask_predict, axis=3)[0,:,:]

    custom_colormap = {0: 0, 1: 100, 2: 150, 3: 200}
    # Áp dụng ánh xạ colormap lên image_prediction
    binary_image = np.vectorize(custom_colormap.get)(image_prediction)
    binary_image = np.clip(binary_image, 0, 255).astype(np.uint8)

    # Lấy các key từ colormap để xử lý
    unique_values = list(custom_colormap.values())[1:]
    key_points=[]
    key=None
    ###### RGB
    # custom_colormap = {
    #     (0, 0, 0): "Background",  # Nền đen
    #     (255, 0, 0): "Class1",    # Màu đỏ
    #     (0, 255, 0): "Class2",    # Màu xanh lá
    #     (0, 0, 255): "Class3",    # Màu xanh dương
    # }
    # unique_values = list(custom_colormap.keys())[1:]  # Loại bỏ lớp nền
    # key_points = []
    # contour_points = []

    # for value in unique_values:
    #     print(f"Processing color: {value}")
    #     # Tạo mặt nạ nhị phân để giữ lại lớp có màu chính xác bằng value
    #     mask = cv2.inRange(image, np.array(value), np.array(value))

    #     # Tìm các contour trong mặt nạ
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ###################3#################


    for value in unique_values:
        print(f'Gia tri value:{value}')
        # Tạo mặt nạ nhị phân, giữ lại các pixel có giá trị chính xác bằng value, những pixel khác về 0
        mask = np.where(binary_image == value, binary_image, 0)

        # Tạo mặt nạ nhị phân để xác định các contour (giá trị khác 0 sẽ được giữ lại, còn lại là 0)
        _, binary_mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

        # Tìm các contour trong vùng này
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour_part in range(len(contours)):
            area=cv2.contourArea(contours[contour_part])
            if area>120 and area<1000000:
                cv2.drawContours(image,contours[contour_part],-1,(255,0,0),2)
                esp= 0.02*cv2.arcLength(contours[contour_part],True)
                approx=(cv2.approxPolyDP(contours[contour_part],esp, True))
                if value== 100:
                    key= 'Class1'
                elif value== 150:
                    key= 'Class2'
                else:
                    key= "NG"
                key_points.append(key)
                contour_points.append(approx)
    return contour_points,key_points