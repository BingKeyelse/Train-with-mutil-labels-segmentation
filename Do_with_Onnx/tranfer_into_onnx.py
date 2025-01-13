import tf2onnx
import tensorflow as tf
import onnx

# Load model
model = tf.keras.models.load_model("checkpoint.hdf5", compile=False)

# Path to save ONNX model
onnx_model_path = "model.onnx"

# Define input signature
spec = (tf.TensorSpec((None, 416, 416, 3), tf.float32, name="input"),)

# Convert TensorFlow model to ONNX model
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the converted ONNX model
onnx.save_model(onnx_model,onnx_model_path)
