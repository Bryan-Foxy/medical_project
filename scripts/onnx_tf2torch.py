import torch
import argparse
import tensorflow as tf
import tf2onnx
from onnx2pytorch import ConvertModel
import os

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Convert a TensorFlow model to PyTorch through ONNX")
    parser.add_argument('--path_model', type=str, required=True, help='Path of the TensorFlow/Keras model to convert (.h5 file)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the converted PyTorch model')
    parser.add_argument('--input_shape', type=str, default="1,256,256,3", help="Input shape for the model, e.g., '1,224,224,3'")

    args = parser.parse_args()

    # Load TensorFlow model
    print(f"Loading TensorFlow/Keras model from: {args.path_model}")
    tensorflow_model = tf.keras.models.load_model(args.path_model)

    # Define input signature
    input_shape = tuple(map(int, args.input_shape.split(",")))
    input_signature = [tf.TensorSpec([None] + list(input_shape[1:]), tf.float32)]

    # Convert TensorFlow model to ONNX
    print("Converting TensorFlow/Keras model to ONNX format...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        tensorflow_model, 
        input_signature=input_signature, 
        opset=13
    )

    # Convert ONNX model to PyTorch
    print("Converting ONNX model to PyTorch format...")
    torch_model = ConvertModel(onnx_model)

    # Save the PyTorch model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving PyTorch model to: {args.output_path}")
    torch.save(torch_model, args.output_path)

    print("Model conversion complete.")

if __name__ == '__main__':
    main()