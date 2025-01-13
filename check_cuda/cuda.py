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

gc.collect()
# your code
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("aaaaffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",iou_score)
# Dinh nghia bien


w, h = 416, 416
batch_size = 32

# Dataset va Dataloader

BACKBONE = "resnet18"
preprocess_input = sm.get_preprocessing(BACKBONE)
metrics11 = [sm.metrics.IOUScore(threshold=0.2), sm.metrics.FScore(threshold=0.2)]

#### them
#config=tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True
#config.log_device_placement= True
#session=tf.compat.v1.Session(config=config)
#### tf.config.experimental.set_memory_growth
#tf.compat.v1.keras.backend.set_session(session)

# Khoi tao model
opt=tf.keras.optimizers.Adam(0.001)
model= Unet(BACKBONE,encoder_weights="imagenet",classes=1,activation="sigmoid",input_shape=(416,416,3),encoder_freeze=True)
loss1 = sm.losses.categorical_focal_dice_loss
model.compile(optimizer=opt,loss=loss1,metrics=[iou_score])

model.load_weights("checkpoint.hdf5")

image = cv2.imread("data2.jpg",1)
namnam = 0
import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (100, 100)

# fontScale
fontScale = 2
 
# Blue color in BGR
color = (255, 245, 0)

# Line thickness of 2 px
thickness = 2

cap = cv2.VideoCapture('video1.mp4')
time1 = time.time()
cc = 1
while(1):
	
	if(time.time() - time1 > 0.2):
		cc = cc + 1
		print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
		if(cc == 100):
			break
	#	time.sleep(3)
	print(time.time() - time1)
	time1 = time.time()

	#ret, frame = cap.read()
	image = cv2.resize(image, (416, 416))
	image11 = image.copy()
	#print("fps",round(1/(time.time() - time1),2))
	mask_predict = model.predict_on_batch(image[np.newaxis, :, :, :])

	#prediction = (mask_predict[0].numpy())*255
	prediction = mask_predict[0]*255
	
	#prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
	prediction = cv2.convertScaleAbs(prediction)

	#prediction = cv2.medianBlur(prediction, 5)
	ret1,th1 = cv2.threshold(prediction,100,255,cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cv2.imwrite("/home/pronics-super/Desktop/check_cuda/result.jpg",th1)
			 
	# Using cv2.putText() method
	cv2.putText(image11, str(round(1/(time.time() - time1),2)), org, font, 
		   fontScale, color, thickness, cv2.LINE_AA)
	cv2.imshow("im",image11)
	print("fps",round(1/(time.time() - time1),2),"error :",str(cc))









