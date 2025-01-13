# Setup and run code with ONNX to reduce model for interfacing with hardware
[Source code ONNX](https://onnxruntime.ai/docs/)

----
### Setup TensorRT
    sudo apt-get update
    sudo apt-get install -y python3-libnvinfer libnvinfer-dev libnvinfer-plugin-dev
### Setup TensorRt python package
    pip install nvidia-pyindex
    pip install nvidia-tensorrt
    
    pip install tf2onnx
    pip install skl2onnx
    pip install onnxruntime-gpu

- Run file **Do_with_Onnx\tranfer_into_onnx.py** to convert weight .hdf5 to .onnx. After that run **Do_with_Onnx\inference_with_onnx.py** to run program for predicting 5 images at once with 24 (FPS - run on P100 Tesla) 

---
# Run and auto labeling with Segmentation

### Run Export_json_file_with_multi_Keras 
- Run the file load_file_json_for_all_file.py to read the image and create a json file to convert to label to read and edit with Labelimg
---
### Run Train_mutil_class_of_Keras
- Run the file main_train_keras.py to training
---
### Export json with Unet
- **Export_json_with_multi_Unet** use for auto labeling with model including mutil class for segmentation and **Export_json_with_singal_class_Unet** run with singal class segmentation
- **Reference link**:
  - [Sử dụng Unet để phân vùng, phát hiện sản phẩm lỗi](https://www.youtube.com/watch?v=lgV1O4fHg6A&t=1783s)
  - [Multiclass semantic segmentation using U-Net](https://www.youtube.com/watch?v=TkngQCI88CA)
  - [Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus/)
  - [Multiclass Segmentation on Crowd Instance-level Human Parsing (CHIP) Dataset using UNET](https://github.com/nikhilroxtomar/Multiclass-Segmentation-on-Crowd-Instance-level-Human-Parsing-CHIP-Dataset-using-UNET/tree/main)