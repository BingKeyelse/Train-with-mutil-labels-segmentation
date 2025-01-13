import os
import random
import cv2 
import shutil

# Thư mục nguồn
root_folder = r"D:\tron_file_imgae_and_json"
root_folder_chua_file=r"D:\chua_file_tron_image_and_json"
selected_files = []
number_new_folder=1


def choose_random_file(root_folder):
        # Lấy danh sách tất cả các tệp trong thư mục gốc và các thư mục con
        all_files = []
        for folder_name, _, file_names in os.walk(root_folder):
            for file_name in file_names:
                if file_name.lower().endswith('.png'):  # Chỉ chọn các tệp có đuôi là ".png"
                    file_path = os.path.join(folder_name, file_name)
                    all_files.append(file_path)

        # Loại bỏ các tệp đã được chọn trước đó
        available_files = [file for file in all_files if file not in selected_files]
        
        # Chọn ngẫu nhiên một tệp từ danh sách các tệp
        if available_files:
            random_file = random.choice(available_files)
            return random_file
        else:
            return None  # Trả về None nếu không có tệp nào được chọn
        

while len(selected_files)<100000:

    # Chọn ngẫu nhiên một tệp từ thư mục gốc
    random_selected_file = choose_random_file(root_folder)
    random_selected_file_json = random_selected_file.replace(".png","")+".json"

    # img=cv2.imread(random_selected_file,1)
    # cv2.imshow('Hinh anh',img)

    if random_selected_file:
        print("Tệp được chọn ngẫu nhiên:", random_selected_file)
    #     print("Tệp được json ngẫu nhiên:", random_selected_file_json)
    # else:
    #     print("Không có tệp nào có đuôi là '.png' trong thư mục gốc hoặc các thư mục con.")

    # Tạo dẫn folder 
    path_new_folder_direct=root_folder_chua_file+"/"+"cam"+str(number_new_folder)+"_NG"
    if os.path.exists(path_new_folder_direct):
        files_in_folder = os.listdir(path_new_folder_direct)
        num_files_for_name = len(files_in_folder) 
        file_name_png=path_new_folder_direct+"/"+str(int(num_files_for_name/2))+".png"
        file_name_json=path_new_folder_direct+"/"+str(int(num_files_for_name/2))+".json"

        # tạo nó
        shutil.copyfile(random_selected_file, file_name_png)
        shutil.copyfile(random_selected_file_json, file_name_json)
        selected_files.append(random_selected_file)


        # print("Tệp được tạo ngẫu nhiên:", file_name_png)
        # print("Tệp được tạo json ngẫu nhiên:", file_name_json)

        if len(files_in_folder) ==998:
            number_new_folder=number_new_folder+1

    else:
        print("sai")
        os.mkdir(path_new_folder_direct)