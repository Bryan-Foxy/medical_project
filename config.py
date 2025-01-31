import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "datas/test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
HEART_MODEL_PATH = os.path.join(MODEL_DIR, "heart_segmentation_model.h5")
VERTEBRAE_MODEL_PATH = os.path.join(MODEL_DIR, "vertebrae_yolo.pt")