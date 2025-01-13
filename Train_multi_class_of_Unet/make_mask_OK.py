import os
import shutil

import cv2
import numpy as np
from PIL import Image

from skimage.draw import ellipse

# Dinh nghia bien
data_path = r"/home/ponics-tiny/Desktop/data/dataset"

# Tap map cho cac anh san pham khong bi loi
def make_map_normal(data_path):
    # Loop through normal folder
    for folder in os.listdir(data_path):
        if (folder.endswith("OK")) and (not folder.startswith(".")) and (not folder.endswith("mask")):
            print("*" * 10, folder)
            # Make mask folder
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try:
                shutil.rmtree(mask_folder) # xoá toàn bộ thư mục và các mục bên trong nó 
            except:
                pass

            os.mkdir(mask_folder) # tạo một folder mới 

            # Loop through file in current folder:
            current_folder = os.path.join(data_path, folder)
            for file in os.listdir(current_folder):
                if file.endswith("png"):
                    print(file)
                    # Read image file
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # Make mask file for normal product - it's blank image, no defect
                    mask_image = np.zeros((w, h), dtype=np.uint8)
                    mask_image = Image.fromarray(mask_image)

                    # Save the file
                    mask_image.save(os.path.join(mask_folder, file))

make_map_normal(data_path)