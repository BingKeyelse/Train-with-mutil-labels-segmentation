import os
import shutil

import cv2
import numpy as np
from PIL import Image
import json
from skimage.draw import polygon


# Dinh nghia bien
data_path = r"DATA"
def draw_defect(file, w, h):
    # Lấy file id
    file_json = file.replace(".png", "")+'.json'
    
    # file_json= os.path.join(data_json_path,file_id)
    # print(file_json)

    # Tạo ảnh màu đen
    mask_image = np.zeros((w, h), dtype=np.uint8)

    with open(file_json, 'r') as file:
        data = json.load(file)
    value=0
    
    for shape in data['shapes']:

        if shape['label']=="NG":
            value=200
        elif shape['label']=="CLASS1":
            value=150
        elif shape['label']=="CLASS2":
            value=100
        else:
            value=0
            
    
        points = shape['points']
        x = [point[0] for point in points]
        y = [point[1] for point in points]

        rr,cc= polygon(y,x)
        
        try:
            # Gán các điểm thuộc hình ellipse thành 1
            mask_image[rr, cc] = value
        except:
            # Nếu lỗi chỉ gán các điểm trong ảnh
            rr_n = [min(w - 1, i) for i in rr]
            cc_n = [min(h - 1, i) for i in cc]
            mask_image[rr_n, cc_n] = value
    
    mask_image = np.array(mask_image, dtype=np.uint8)

    mask_image = Image.fromarray(mask_image)


    return mask_image


# Tap map cho cac anh san pham khong bi loi
def make_map_defect(data_path):
    # Loop through normal folder
    for folder in os.listdir(data_path):
        if (not folder.startswith(".")) and (not folder.endswith("mask")) :
        # if (folder.endswith("NG")) and (not folder.startswith(".")) and (not folder.endswith("mask")) :
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

                # Check file trùng
                if file.find("(") > -1:
                    # Xoá file nếu bị trùng (do đặc thù dữ liệu)
                    os.remove(os.path.join(current_folder, file))
                    continue

                if file.endswith("png"):
                    # print(file)
                    # Read image file
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # Make mask file for defect product - it's blank image with defect
                    mask_image = draw_defect(current_file, w, h)

                    # Save the file
                    mask_image.save(os.path.join(mask_folder, file))
    
make_map_defect(data_path)