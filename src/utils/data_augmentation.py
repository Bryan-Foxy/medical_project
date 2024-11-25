import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(height=384, width=384):
    """
    Augmentation pipeline for training.
    Includes a variety of augmentations to improve generalization.
    """
    return A.Compose([
        # Resize to fixed dimensions
        A.Resize(height=height, width=width),

        # Flip and rotation augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Spatial transformations
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.2, p=0.5),

        # Intensity and noise augmentations
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),

        # Crop and cut augmentations
        A.RandomCrop(height=height, width=width, p=0.5),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def get_validation_transforms(height=384, width=384):
    """
    Augmentation pipeline for validation.
    Keeps transformations simple and consistent.
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})