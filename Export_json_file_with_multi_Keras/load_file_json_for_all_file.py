import os
import shutil

import cv2
import numpy as np
from PIL import Image
import json
from skimage.draw import polygon
from code_create_json_all_of_file import*

data_path = 'picture'

current_folder='picture'
print("*" * 10, current_folder)
for file in os.listdir(current_folder):

    if file.find("(") > -1:
        # Xoá file nếu bị trùng (do đặc thù dữ liệu)
        os.remove(os.path.join(current_folder, file))
        continue

    if file.endswith("png"):
        image_path = os.path.join(current_folder, file)
        file_json = image_path.replace(".png", "")+'.json'
        # with open(file_json, "x"):
        #     pass  # không cần ghi bất kỳ dữ liệu nào, chỉ cần đóng tệp lại
        # print(file_json)
        contour_points,key_points= Unet_detect_position(image_path)
        float_array_list = convert_to_float(contour_points)
        # Chuyển đổi các điểm contour
        converted_contours = []
        for contour in float_array_list:
            converted_contour = convert_contour_coordinates(contour, old_shape, new_shape)
            converted_contours.append(converted_contour)

        create_labelme_json(image_path, converted_contours, file_json,file,key_points)
