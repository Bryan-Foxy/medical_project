import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Define transformations for training data.
    Includes augmentations like resizing, flipping, rotation, and normalization.
    """
    return A.Compose([
        A.Resize(512, 512),  
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),  
        A.VerticalFlip(p=0.5),    
        A.ShiftScaleRotate(
            shift_limit=0.05,    
            scale_limit=0.05,     
            rotate_limit=15,      
            p=0.5                
        ),
        
        # Random Brightness and Contrast adjustments
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  
            contrast_limit=0.2,    
            p=0.3                  
        ),
        
        # Blur or sharpen the image slightly
        A.GaussianBlur(blur_limit=(3, 7), p=0.2), 
        A.MotionBlur(blur_limit=3, p=0.1),        
        
        # Elastic deformation to simulate realistic distortions
        A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50, p=0.2
        ),

        # Normalize the image to match pre-trained model input expectations
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        # Convert to PyTorch tensor
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_validation_transforms():
    """
    Define transformations for validation data.
    Focus on resizing and normalization without augmentation.
    """
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})