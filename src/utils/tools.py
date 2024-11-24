import os
import torch
import matplotlib.pyplot as plt

def get_device():
	""" Return device to computation for neural network"""
	if torch.backends.mps.is_available() and torch.backends.mps.is_built():
	    device = torch.device("mps")
	    return device
	elif torch.cuda.is_available():
	    device = torch.device("cuda")
	    return device
	else:
	    device = torch.device("cpu")
	    return device

def plot_losses(train_losses, val_losses, save_path="../saves/images/loss_unet_curve.png"):
    """
    Plot and save the training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        save_path (str): Path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linestyle='-', marker='o', color='blue')
    plt.plot(val_losses, label="Validation Loss", linestyle='--', marker='s', color='orange')
    plt.title("Loss Curve Across Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Loss curve saved at: {save_path}")
    plt.close()