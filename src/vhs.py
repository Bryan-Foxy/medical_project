import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from config import IMAGES_OUTPUT
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class VHS:
    def __init__(self, image_path, model_heart, model_vertebrae):
        """
        Initializer for the VHS class.
        
        Parameters:
        - image_path (str): Path to the input image for processing.
        - model_heart (str): Path to the heart segmentation model.
        - model_vertebrae (str): Path to the vertebrae detection model (YOLO).
        """
        self.model_heart = tf.keras.models.load_model(model_heart)
        self.model_vertebrae = YOLO(model_vertebrae)
        self.image_path = image_path
        self.img_size = (256, 256)  # Resize for segmentation model

    def _resize_image(self, image, original_shape):
        """Resize a binary mask back to the original image dimensions."""
        resized = cv2.resize(image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized

    def _predictions(self):
        """
        Perform predictions to detect the heart segmentation mask and vertebrae bounding boxes.

        Returns:
        - predicted_heart_mask_resized (ndarray): Resized heart segmentation mask.
        - predicted_vertebrae (YOLO predictions): Detected vertebrae bounding boxes.
        - original_image (ndarray): The original image.
        """
        # Load and preprocess the image for heart segmentation
        preprocess_image = load_img(self.image_path, target_size=self.img_size)
        preprocess_image = img_to_array(preprocess_image) / 255.0
        preprocess_image = np.expand_dims(preprocess_image, axis=0)

        # Predict heart segmentation mask
        prediction = self.model_heart.predict(preprocess_image)
        predicted_mask = (prediction > 0.5).astype(np.uint8)
        predicted_heart_mask = predicted_mask[0, :, :, 0] * 255

        # Resize heart mask back to original image size
        original_image = cv2.imread(self.image_path)
        predicted_heart_mask_resized = self._resize_image(predicted_heart_mask, original_image.shape)

        # Predict vertebrae bounding boxes using YOLO
        predicted_vertebrae = self.model_vertebrae(self.image_path)

        return predicted_heart_mask_resized, predicted_vertebrae, original_image

    def _calculate_heart_diameters(self, mask):
        """
        Calculate the major (long) and minor (short) diameters of the heart from the segmented mask using PCA.

        Parameters:
        - mask (ndarray): The heart segmentation mask.

        Returns:
        - major_diameter (float): The major (long) diameter of the heart.
        - minor_diameter (float): The minor (short) diameter of the heart.
        - major_p1, major_p2 (tuple): The endpoints of the major diameter.
        - minor_p1, minor_p2 (tuple): The endpoints of the minor diameter.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the heart mask.")

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        points = np.squeeze(largest_contour)

        # Center the points
        mean = np.mean(points, axis=0)
        centered_points = points - mean

        # Perform PCA
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Major (long) and minor (short) axes
        major_vector = eigenvectors[:, 0]  # First principal component (long axis)
        minor_vector = eigenvectors[:, 1]  # Second principal component (short axis)

        # Project points onto the major and minor axes
        major_projection = np.dot(centered_points, major_vector)
        minor_projection = np.dot(centered_points, minor_vector)

        # Calculate endpoints of the major and minor axes
        major_p1 = mean + major_vector * np.min(major_projection)
        major_p2 = mean + major_vector * np.max(major_projection)
        minor_p1 = mean + minor_vector * np.min(minor_projection)
        minor_p2 = mean + minor_vector * np.max(minor_projection)

        # Calculate lengths of the axes
        major_diameter = np.linalg.norm(major_p2 - major_p1)
        minor_diameter = np.linalg.norm(minor_p2 - minor_p1)

        return major_diameter, minor_diameter, tuple(major_p1.astype(int)), tuple(major_p2.astype(int)), tuple(minor_p1.astype(int)), tuple(minor_p2.astype(int))

    def _calculate_vertebrae_lengths(self, vertebrae_predictions, scale_factor):
        """Calculate the lengths of the detected vertebrae."""
        lengths = []
        for box in vertebrae_predictions[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            length = (y2 - y1) * scale_factor
            lengths.append(length)
        return sorted(lengths, reverse=True)

    def _calculate_vhs_score(self, major_diameter, minor_diameter, vertebra_lengths):
        """Calculate the VHS score."""
        def superpose_diameter(diameter, lengths):
            count = 0
            for length in lengths:
                if diameter >= length:
                    count += 1
                    diameter -= length
                    if diameter <= 0:
                        break
                else:
                    count += round(diameter / length, 1)
                    break
            return count

        count_major = superpose_diameter(major_diameter, vertebra_lengths)
        count_minor = superpose_diameter(minor_diameter, vertebra_lengths)
        return count_major, count_minor

    def perform_vhs(self, scale_factor=0.3528):
        """
        Perform the full VHS calculation and optionally visualize the results.

        Parameters:
        - scale_factor (float): Scale factor for converting pixel distances to real-world measurements (default 0.3528).

        Returns:
        - vhs_score (float): The calculated VHS score.
        - output_path (str): Path to the output visualization image.
        """
        # Perform predictions
        heart_mask, vertebrae_predictions, original_image = self._predictions()

        # Calculate heart diameters
        major_diameter, minor_diameter, major_p1, major_p2, minor_p1, minor_p2 = self._calculate_heart_diameters(heart_mask)
        major_diameter *= scale_factor
        minor_diameter *= scale_factor

        # Calculate vertebrae lengths
        vertebra_lengths = self._calculate_vertebrae_lengths(vertebrae_predictions, scale_factor)

        # Calculate VHS score
        count_major, count_minor = self._calculate_vhs_score(major_diameter, minor_diameter, vertebra_lengths)
        vhs_score = count_major + count_minor

        # Visualization
        output_name = f"{os.path.splitext(os.path.basename(self.image_path))[0]}_vhs.png"
        output_path = os.path.join("output", output_name)
        os.makedirs("output", exist_ok=True)

        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        cv2.arrowedLine(image_rgb, major_p1, major_p2, (255, 0, 0), 2, tipLength=0.05)  # Long axis (red)
        cv2.arrowedLine(image_rgb, minor_p1, minor_p2, (0, 255, 0), 2, tipLength=0.05)  # Short axis (green)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_rgb, f'L: {major_diameter:.1f} mm', major_p1, font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image_rgb, f'S: {minor_diameter:.1f} mm', minor_p1, font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        vhs_score_text = f'VHS = {count_major} + {count_minor} = {vhs_score:.1f}'
        cv2.putText(image_rgb, vhs_score_text, (50, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        return (round(major_diameter, 2), count_major), (round(minor_diameter, 2), count_minor), vhs_score, output_path