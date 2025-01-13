import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import os
import random

import tensorflow as tf
import cv2
import numpy as np
import time
# Import thu vien segmentation_models
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
sm.set_framework("tf.keras")
sm.framework()
import tensorflow as tf
import gc
from keras import backend as K
import torch

# Cấu hình TensorFlow và thu gom bộ nhớ
os.environ["SM_FRAMEWORK"] = "tf.keras"
gc.collect()
sm.set_framework("tf.keras")

# Kiểm tra GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Kích thước ảnh đầu vào
w, h = 416, 416

# Khởi tạo mô hình
BACKBONE = "resnet18"
metrics = [sm.metrics.IOUScore(threshold=0.2), sm.metrics.FScore(threshold=0.2)]
preprocess_input = sm.get_preprocessing(BACKBONE)

model = Unet(
    BACKBONE,
    encoder_weights="imagenet",
    classes=1,
    activation="sigmoid",
    input_shape=(h, w, 3),
    encoder_freeze=True,
)
loss = sm.losses.categorical_focal_dice_loss
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=loss, metrics=[iou_score])
model.load_weights("checkpoint.hdf5")
model.summary()

# Danh sách các ảnh
image_paths1 = ["data1.jpg", "data2.jpg", "data3.jpg", "data4.jpg"]

image_paths = image_paths1 

# Vòng lặp dự đoán
while True:
    time_start = time.time()

    # Tải và preprocess batch ảnh
    batch_images = []
    for image_path in image_paths:
        img = cv2.imread(image_path, 1)
        img = cv2.resize(img, (w, h))  # Resize về 416x416
        batch_images.append(img)
    batch_images = np.array(batch_images)

    # Dự đoán cho cả batch
    # mask_predictions = model.predict_on_batch(batch_images, batch_size=len(image_paths))
    mask_predictions = model.predict_on_batch(batch_images)

    # Xử lý từng mask
    for i, mask_prediction in enumerate(mask_predictions):
        prediction = (mask_prediction[:, :, 0] * 255).astype(np.uint8)  # Dạng 8-bit

        # Tạo binary mask từ threshold
        _, binary_mask = cv2.threshold(prediction, 100, 255, cv2.THRESH_BINARY)

        # Lưu ảnh kết quả
        result_path = f"/home/pronics-super/Desktop/check_cuda/result_normal{i}.jpg"
        cv2.imwrite(result_path, binary_mask)

        # Tìm contours (nếu cần thêm phân tích)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(f"Saved result for image {image_paths[i]} -> {result_path}")
        # cv2.imshow(f'So thu tu {i}',binary_mask)
        

    # Tính FPS
    fps = 1 / (time.time() - time_start)
    print(f"FPS: {fps:.2f}")
    # cv2.waitKey(1)