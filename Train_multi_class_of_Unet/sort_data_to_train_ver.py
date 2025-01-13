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

label_encoder= LabelEncoder()
mapping=[0 ,100 ,150 ,200]
# label_encoder.fit(list(mapping.keys()))
# label_encoder.fit_transform(mapping)
label_encoder.fit_transform(np.array(mapping).ravel())


# Dinh nghia bien
data_path  = "dataset"
# data_path  = "dataset"
w, h = 512, 512
batch_size = 10

# Dataset va Dataloader

BACKBONE = "resnet50"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Dung de tao toan bo du lieu va load theo batch
class Dataset:
    def __init__(self, image_path, mask_path, w, h):
        # the paths of images
        self.image_path = image_path
        # the paths of segmentation images
        self.mask_path = mask_path

        self.w = w
        self.h = h

    def __getitem__(self, i):
        #Capture mask/label info as a list
        # train_masks = [] 
        # read data
        image = cv2.imread(self.image_path[i])
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        image = preprocess_input(image)

        mask = cv2.imread(self.mask_path[i], cv2.IMREAD_UNCHANGED)
        image_mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        # train_masks = np.array(train_masks)

        # n, h, w = image_mask.shape
        train_masks_flatten= image_mask.reshape(-1,1)

        train_masks_flatten_encoded= label_encoder.transform(train_masks_flatten.ravel())
    
        mask= train_masks_flatten_encoded.reshape(self.w,self.h)

        # np.unique(mask)
        # image_mask = np.expand_dims(mask, axis=3)
        image_mask=to_categorical(mask,num_classes=4)
        # Add a channel dimension to the mask
        image_mask = np.expand_dims(image_mask, axis=-1)
        # image_mask= image_mask.reshape(image_mask.shape[0], image_mask.shape[1],4)
        return image, image_mask

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size,shape, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.indexes = np.arange(self.shape)

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)
        # return data

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

#  Load thong tin tu folder dataset de tao 2 bien image_path, mask_path
def load_path(data_path):
    # Get normal image and mask

    # classes = ['human', 'dog', 'cat']
    classes = ['pic']
    # Lop qua cac thu muc khong loi
    normal_image_path = []
    normal_mask_path = []
    for class_ in classes:
        current_folder = os.path.join(data_path, class_)
        for file in os.listdir(current_folder):
            if file.endswith("png") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file) 
                mask_path = os.path.join(current_folder + "_mask", file)
                normal_mask_path.append(mask_path)
                normal_image_path.append(image_path)

    image_path = normal_image_path
    mask_path = normal_mask_path

    return image_path, mask_path

# Thu hien load va train model

# Load duong dan vao 2 bien
image_path, mask_path = load_path(data_path)

# Chia du lieu train, test van o dang path
image_train, image_test, mask_train, mask_test = train_test_split(image_path, mask_path, test_size=0.3)

# Tao dataset va dataloader
train_dataset = Dataset(image_train, mask_train, w, h)
test_dataset = Dataset(image_test, mask_test, w, h)

train_loader = Dataloader(train_dataset, batch_size, shape=len(image_train), shuffle=True)
test_loader = Dataloader(test_dataset, batch_size, shape=len(image_test), shuffle=True)
print(train_loader.shape)

# Khoi tao model
opt=tf.keras.optimizers.Adam(0.0001)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model= Unet(BACKBONE,encoder_weights='imagenet',classes=4,activation="softmax",encoder_freeze=True)
model.compile(opt, total_loss, metrics=metrics)

# print(model.summary())

class_colors = {
    0: (0,0,0),    # zero
    1: (0,0,255),    # cat xanh
    2: (255,0,0),    # red dog
    3: (255,255,255)   # vang human
    # Add more colors for additional classes if needed
}

signal_colors = {
    0: 0,       # Label 0 -> Màu 0
    1: 100,     # Label 1 -> Màu 100
    2: 150,     # Label 2 -> Màu 150
    3: 200      # Label 3 -> Màu 200
}

# Train model
# is_train = True
is_train = False
if is_train:
    from keras.callbacks import ModelCheckpoint
    # filepath_wights="haha_ver2_hhhhhh.hdf5"
    # model.load_weights(filepath_wights)
    filepath= "chekcheck.hdf5"
    # callback = ModelCheckpoint(filepath, monitor='val_f-score', verbose=1, save_best_only=True,mode='max')
    # callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
    callback = ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True,mode='max')

    # model.fit_generator( train_loader, validation_data=test_loader, epochs=200, callbacks=[callback])
    model.fit( train_loader, validation_data=test_loader, epochs=30, callbacks=[callback])

else:
    # Load model de test
    model.load_weights("chekcheck.hdf5")

    ids = range(len(image_test))
    index = random.sample(ids,15)

    import matplotlib.pyplot as plt

    for id in index:

        # Anh dau vao, ko phai mask
        image = cv2.imread(image_test[id],1)
        image = cv2.resize(image, (512, 512))
        image_use=  np.expand_dims(image, 0)
        print(image_use.shape)
        # print(image)
        image_use= preprocess_input(image_use)
        # Dua qua model de predicted segmentation map

        mask_predict = model.predict(image_use)
        print(mask_predict)
        image_prediction = np.argmax(mask_predict, axis=3)[0,:,:]
        print(np.max(image_prediction))
        # print(image_prediction.shape)

        ## detect label has a max area
        area1= np.sum(image_prediction==1)
        area2= np.sum(image_prediction==2)
        area3= np.sum(image_prediction==3)
        print('dien tich tung nhan la: 1 la {} , 2 la {},  3 la {}'.format(area1,  area2, area3))

        ###### Anh gray
        custom_colormap = {0: 0, 1: 100, 2: 150, 3: 200}
        # Áp dụng ánh xạ colormap lên image_prediction
        binary_image = np.vectorize(custom_colormap.get)(image_prediction)
        # binary_image = np.where(modified_image > 0, 255, 0).astype(np.uint8)

        # Tạo một danh sách chứa các diện tích và nhãn tương ứng
        areas = [area1, area2, area3]
        labels = [0, 1, 2, 3]

        # Xác định nhãn có diện tích lớn nhất
        max_area_label = labels[np.argmax(areas)]
        print(max_area_label)


        with open('pixel_values.txt', 'w') as file:
        # # Iterate through each row
            for row in range(image_prediction.shape[0]):
                # Concatenate the pixel values of the row into a single string
                row_values = ' '.join(map(str, image_prediction[row]))
                # Write the concatenated string to the file
                file.write(row_values + '\n')
        
        #  Create color image for prediction
        color_image = np.zeros_like(image)
        for label, color in class_colors.items():
            color_image[image_prediction == label] = color
        # color_image=np.array(color_image)

        overlay = cv2.addWeighted(image, 0.6, color_image, 0.4, 0)

        # Doc mask thuc te
        image_mask = cv2.imread(mask_test[id], cv2.IMREAD_UNCHANGED)
        
        image_mask = cv2.resize(image_mask, (512, 512))

        plt.figure(figsize=(10, 6))
        plt.subplot(221)
        plt.title("Hình ảnh sản phẩm")
        plt.imshow(image) 

        plt.subplot(222)
        plt.title("Vết dự đoán")
        plt.imshow(overlay)

        plt.subplot(223)
        plt.title("Vết lỗi thật sự")
        # z = image_prediction[0]#[:, :, 0]
        # plt.imshow(color_image, cmap='gray')
        plt.imshow(image_mask)

        plt.subplot(224)
        plt.title("Vết lỗi maskkkkkk binary")
        plt.imshow(binary_image)
        
        plt.show()
        










