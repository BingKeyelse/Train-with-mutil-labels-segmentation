import json
from json import JSONEncoder
from code_unet_to_import_to_create_json_all_of_files import Unet_detect_position
import os
import numpy as np
import base64
import io
import numpy as np
from PIL import Image

old_shape = (512, 512)
new_shape = (960, 640)

def img_to_b64(image_path):
    # Đọc hình ảnh từ đường dẫn
    img_pil = Image.open(image_path)
    
    # Chuyển đổi hình ảnh thành mảng NumPy
    img_arr = np.array(img_pil)
    
    # Chuyển đổi mảng NumPy thành dữ liệu Base64
    img_b64 = str(img_arr_to_b64(img_arr))
    
    return img_b64

def img_arr_to_b64(img_arr):
    img_pil = Image.fromarray(img_arr)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    img_b64 = base64.b64encode(img_byte_arr).decode('utf-8')
    return img_b64

def convert_to_float(array_list):
    float_array_list = []
    for array in array_list:
        # Chuyển đổi từ int sang float
        float_array = array.astype(float)
        float_array_list.append(float_array)
    return float_array_list

def convert_contour_coordinates(contour, old_shape, new_shape):
    old_height, old_width = old_shape
    new_height, new_width = new_shape

    # Tính toán hệ số tỷ lệ
    scale_x = new_width / old_width
    scale_y = new_height / old_height

    # Chuyển đổi các tọa độ contour
    converted_contour = []
    for point in contour:
        x, y = point[0]
        new_x = x * scale_x
        new_y = y * scale_y
        converted_contour.append([[new_x, new_y]])

    return np.array(converted_contour)

def change_array(input):
    array_2d = []
    # input.astype(float).reshape(-1, 2)
    for subarray in input:
        for point in subarray:
            array_2d.append(point.tolist())
    # array_2d = np.array(array_2d)  # Chuyển danh sách thành mảng NumPy
    # array_2d = array_2d.astype(float).reshape(-1, 2)
    return array_2d

def create_labelme_json(image_path, contour_points, save_path,name_image):
    labelme_data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [],
        "imagePath": str(name_image),
        "imageData": img_to_b64(image_path),
        "imageHeight": 960,  # Cần cập nhật kích thước hình ảnh thực tế
        "imageWidth": 640    # Cần cập nhật kích thước hình ảnh thực tế
    }

    # Tạo dữ liệu shape từ các điểm contour
    for contour in contour_points:
        shape = {
            "label": "NG",
            "points": change_array(contour),
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        labelme_data["shapes"].append(shape)

    # Lưu dữ liệu vào tệp JSON
    with open(save_path, "w") as json_file:
        json.dump(labelme_data, json_file, indent=2)

