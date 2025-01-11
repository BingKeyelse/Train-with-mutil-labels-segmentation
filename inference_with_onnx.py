# import onnx
# onnx_model = onnx.load("model.onnx")
# onnx.checker.check_model(onnx_model) ## Check


import os
import time
import cv2
import numpy as np
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    raise ImportError("The 'onnxruntime' module is not installed. Please install it with 'pip install onnxruntime' before running this script.")

# Check GPU support for ONNX Runtime
print("Available providers:", ort.get_available_providers())
if 'CUDAExecutionProvider' in ort.get_available_providers():
    ort_sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
else:
    ort_sess = ort.InferenceSession("model.onnx")

# Input image dimensions
w, h = 416, 416

# List of image paths
image_paths = ["data1.jpg", "data2.jpg", "data3.jpg", "data4.jpg", "data5.jpg"] ## Link file image

# Prediction loop
while True:
    time_start = time.time()

    # Load and preprocess batch images
    batch_images = []
    for image_path in image_paths:
        img = cv2.imread(image_path, 1)
        if img is None:
            print(f"Error: Could not read image '{image_path}'. Skipping...")
            continue
        img = cv2.resize(img, (w, h))  # Resize to 416x416
        batch_images.append(img)
    if not batch_images:
        print("No valid images to process. Exiting loop.")
        break

    # Convert batch images to a numpy array and preprocess for ONNX model
    batch_images = np.array(batch_images, dtype=np.float32)  # Ensure dtype is float32
    # batch_images = np.transpose(batch_images, (0, 1,2,3))  # Convert to (batch_size, channels, height, width)
    # batch_images /= 255.0  # Normalize to [0, 1]

    # Run inference with ONNX Runtime
    outputs = ort_sess.run(None, {"input": batch_images})[0]

    # Process each output mask
    for i, mask_prediction in enumerate(outputs):
        mask_prediction = (mask_prediction[:,:,0] * 255).astype(np.uint8)  # Take the first channel and convert to 8-bit

        # Resize mask back to original dimensions if needed
        binary_mask = cv2.resize(mask_prediction, (w, h))

        # Create a binary mask with thresholding
        _, binary_mask = cv2.threshold(binary_mask, 20, 255, cv2.THRESH_BINARY)

        # Save the result image
        result_path = f"/home/pronics-super/Desktop/check_cuda/result_onnx{i}.jpg"
        cv2.imwrite(result_path, binary_mask)

        # Optionally find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(f"Saved result for image {image_paths[i]} -> {result_path}")

    # Calculate FPS
    fps = 1 / (time.time() - time_start)
    print(f"FPS: {fps:.2f}")
