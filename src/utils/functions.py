import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp

class DiceLoss(torch.nn.Module):
    def forward(self, predictions, targets):
        smooth = 1.0
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        return 1 - dice

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def load_heart_model(model_path, device):
    """
    Load heart segmentation model.
    """
    # Recreate the model architecture
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None,          
        in_channels=3,                 
        classes=1                    
    )

    # Load the state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()
    return model.to(device)

def plot_heart_predicted(image, mask_predicted, threshold = 0.5):
    # Binarize the predicted mask
    image = np.array(Image.fromarray(image).resize(mask_predicted.shape))
    binary_mask = (mask_predicted > threshold).astype(np.uint8)

    # Create a visualization by overlaying the mask on the image
    overlay = np.zeros_like(image)
    overlay[..., 1] = binary_mask * 255  # Green channel for mask
    overlay_image = (0.6 * image + 0.4 * overlay).astype(np.uint8)  # Blend original image with mask

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask_predicted, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay (Image + Mask)")
    plt.imshow(overlay_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def is_image_file(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return filename.lower().endswith(valid_extensions)

def prediction_heart(model, image_path, transforms, device="cpu", visualize=False):
    """
    Predict the mask of the heart from an image or directory of images.

    Args:
        model: The segmentation model.
        image_path: Path to an image file or directory containing images.
        transforms: Albumentations transformation function.
        device: Device to use for prediction (default: "cpu").
        visualize: Whether to visualize the prediction (default: False).

    Returns:
        Predicted mask for a single image or list of masks for a directory.
    """
    try:
        # If image_path is a single file
        if os.path.isfile(image_path) and is_image_file(image_path):
            image = np.array(Image.open(image_path).convert("RGB"))  # Ensure RGB format
            augmented = transforms(image=image)  # Apply Albumentations transform
            image_tensor = torch.tensor(augmented["image"]).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                pred = model(image_tensor)
                pred = torch.sigmoid(pred)  # Convert logits to probabilities
                pred = pred.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
            
            # Visualize if requested
            if visualize:
                plot_heart_predicted(image, pred, threshold=0.5)

            return pred

        # If image_path is a directory
        elif os.path.isdir(image_path):
            predictions = []
            for filename in tqdm(os.listdir(image_path), desc="Processing images"):
                full_path = os.path.join(image_path, filename)
                if os.path.isfile(full_path) and is_image_file(full_path):
                    image = np.array(Image.open(full_path).convert("RGB"))
                    augmented = transforms(image=image)
                    image_tensor = torch.tensor(augmented["image"]).unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred = model(image_tensor)
                        pred = torch.sigmoid(pred)
                        pred = pred.squeeze(0).squeeze(0).cpu().numpy()

                    predictions.append((filename, pred))

            return predictions

        else:
            raise ValueError(f"Invalid path: {image_path}")

    except Exception as e:
        print(f"An error occurred: {e}")