import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Librairies
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from src.utils.tools import get_device, plot_losses
from src.utils.dataset_coco import HeartSegmentationDataset
from src.utils.functions import DiceLoss, EarlyStopping, calculate_iou
from src.utils.data_augmentation import get_train_transforms, get_validation_transforms


def load_data(train_json_path, train_images_dir, val_json_path, val_images_dir, batch_size):
    """
    Load training and validation data with separate paths for validation.
    """
    train_dataset = HeartSegmentationDataset(
        json_path=train_json_path,
        images_dir=train_images_dir,
        transform=get_train_transforms()
    )

    val_dataset = HeartSegmentationDataset(
        json_path=val_json_path,
        images_dir=val_images_dir,
        transform=get_validation_transforms()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if device != "cpu" else False,
        num_workers=0 if device == "mps" else 2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True if device != "cpu" else False,
        num_workers=0 if device == "mps" else 2
    )

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """
    Enhanced training function with IoU metric, early stopping, and mixed precision.
    """
    best_iou = 0.0
    losses_train = []
    losses_valid = []
    losses_train_iou = []
    losses_valid_iou = []
    early_stopping = EarlyStopping(patience=5)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks.unsqueeze(1))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                loss.backward()
                optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            train_iou += calculate_iou(torch.sigmoid(outputs), masks.unsqueeze(1)).item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{train_iou/(progress_bar.n+1):.4f}"
            })
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation phase
        val_loss, val_iou = validate_model(model, val_loader, criterion)

        # Store in list
        losses_train.append(train_loss)
        losses_valid.append(val_loss)
        losses_train_iou.append(train_iou)
        losses_valid_iou.append(val_iou)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, "best_unet_model.pth")
            print("Model saved!")
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Plot and store images
    plot_losses(losses_train, losses_valid, save_path = "../saves/images/loss_curve.png")
    plot_losses(losses_train_iou, losses_valid_iou, save_path = "../saves/images/loss_iou_curve.png")


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            
            val_loss += loss.item()
            val_iou += calculate_iou(torch.sigmoid(outputs), masks.unsqueeze(1)).item()
    
    return val_loss / len(val_loader), val_iou / len(val_loader)

if __name__ == "__main__":
    # Paths and hyperparameters
    train_json_path = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/processed datas/veterinary_heart_vert_detection.v1i.coco/train/_annotations.coco.json"
    train_images_dir = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/processed datas/veterinary_heart_vert_detection.v1i.coco/train"
    val_json_path = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/processed datas/veterinary_heart_vert_detection.v1i.coco/valid/_annotations.coco.json"
    val_images_dir = "/Users/armandbryan/Documents/aivancity/PGE5/Medical AI/processed datas/veterinary_heart_vert_detection.v1i.coco/valid"
    
    batch_size = 16
    learning_rate = 2e-4
    num_epochs = 100
    device = get_device()
    
    # Load data
    train_loader, val_loader = load_data(
        train_json_path, train_images_dir,
        val_json_path, val_images_dir,
        batch_size
    )
    
    # Initialize model and training components
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    
    # Combined loss function
    criterion = torch.nn.ModuleList([
        torch.nn.BCEWithLogitsLoss(),
        DiceLoss()
    ])
    
    def combined_loss(pred, target):
        return criterion[0](pred, target) + criterion[1](pred, target)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train and validate the model
    train_model(model, train_loader, val_loader, combined_loss, optimizer, scheduler, num_epochs)