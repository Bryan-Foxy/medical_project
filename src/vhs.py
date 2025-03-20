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
        predicted_vertebrae = self.model_vertebrae.predict(self.image_path)

        return predicted_heart_mask_resized, predicted_vertebrae, original_image

    def _calculate_heart_diameters(self, mask):
        """
        Calculate the major and minor diameters of the heart from the segmented mask.

        Parameters:
        - mask (ndarray): The heart segmentation mask.

        Returns:
        - max_distance (float): The major diameter of the heart.
        - minor_radius (float): The minor diameter of the heart.
        - major_p1, major_p2 (ndarray): The endpoints of the major diameter.
        - minor_p1, minor_p2 (ndarray): The endpoints of the minor diameter.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No contours found in the heart mask.")

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        points = np.squeeze(largest_contour)

        # Calculate major diameter
        max_distance = 0
        major_p1, major_p2 = None, None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_distance:
                    max_distance = dist
                    major_p1, major_p2 = points[i], points[j]

        # Calculate minor diameter
        center = (major_p1 + major_p2) / 2
        major_direction = (major_p2 - major_p1) / np.linalg.norm(major_p2 - major_p1)
        minor_direction = np.array([-major_direction[1], major_direction[0]])

        projections = np.dot(points - center, minor_direction)
        minor_radius = np.max(projections) - np.min(projections)
        minor_p1 = center + minor_direction * np.min(projections)
        minor_p2 = center + minor_direction * np.max(projections)

        return max_distance, minor_radius, major_p1, major_p2, minor_p1, minor_p2

    def _calculate_vertebrae_lengths(self, vertebrae_predictions, scale_factor):
        """Calculate the lengths of the detected vertebrae."""
        lengths = []
        for box in vertebrae_predictions[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            length = (y2 - y1) * scale_factor
            lengths.append(length)
        return sorted(lengths, reverse=True)

    def calculate_vhs(self, major_diameter, minor_diameter, vertebra_lengths):
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
        """
        heart_mask, vertebrae_predictions, original_image = self._predictions()
        major_diameter, minor_diameter, major_p1, major_p2, minor_p1, minor_p2 = self._calculate_heart_diameters(heart_mask)

        major_diameter *= scale_factor
        minor_diameter *= scale_factor
        vertebra_lengths = self._calculate_vertebrae_lengths(vertebrae_predictions, scale_factor)

        # Calculate VHS score
        count_major, count_minor = self.calculate_vhs(major_diameter, minor_diameter, vertebra_lengths)
        vhs_score = count_major + count_minor

        # Convert to int for the vizu
        major_p1 = (int(major_p1[0]), int(major_p1[1]))
        major_p2 = (int(major_p2[0]), int(major_p2[1]))
        minor_p1 = (int(minor_p1[0]), int(minor_p1[1]))
        minor_p2 = (int(minor_p2[0]), int(minor_p2[1]))

        # Visualization
        output_name = f"{os.path.splitext(os.path.basename(self.image_path))[0]}.png"
        output_path = os.path.join(IMAGES_OUTPUT, output_name)
        image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.arrowedLine(image_rgb, tuple(major_p1), tuple(major_p2), (255, 255, 255), 2, tipLength=0.05)
        cv2.arrowedLine(image_rgb, tuple(minor_p1), tuple(minor_p2), (255, 255, 255), 2, tipLength=0.05)
        font = cv2.FONT_HERSHEY_SIMPLEX
        major_text_position = ((major_p1[0] + major_p2[0]) // 2, (major_p1[1] + major_p2[1]) // 2)
        minor_text_position = ((minor_p1[0] + minor_p2[0]) // 2, (minor_p1[1] + minor_p2[1]) // 2)
        cv2.putText(image_rgb, f'{major_diameter:.1f} mm', major_text_position, font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_rgb, f'{minor_diameter:.1f} mm', minor_text_position, font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vhs_score_text = f'IB = {count_major} + {count_minor} = {vhs_score:.1f}'
        cv2.putText(image_rgb, vhs_score_text, (50, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vhs_title_text = f"VHS: {vhs_score:.1f}"
        cv2.putText(image_rgb, vhs_title_text, (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        return (round(major_diameter, 2), count_major), (round(minor_diameter,2), count_minor), vhs_score, output_path
