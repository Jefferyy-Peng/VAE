import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image
from diffusers import AutoencoderKL
from pytorch_lightning import seed_everything
from torchvision import transforms
import matplotlib.pyplot as plt

def center_crop(image):
    """
        Crop the largest center square of the image.

        Args:
        - image (Tensor): The input image as a Tensor of size b, c, h, w.

        Returns:
        - Tensor: The cropped and resized image as a Tensor.
        """
    # Step 1: Determine the size of the largest square
    height, width = image.shape[-2:]
    new_edge = min(height, width)

    # Step 2: Calculate the coordinates for cropping
    left = (width - new_edge) // 2
    top = (height - new_edge) // 2
    right = (width + new_edge) // 2
    bottom = (height + new_edge) // 2

    # Step 3: Crop the image
    image_cropped = image[..., top:bottom, left:right]
    return image_cropped, image_cropped.shape[-2:]

# Define transforms for preprocessing
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # Resize images to (256, 256)
    transforms.ToTensor(),           # Convert images to tensors
])

# Path to COCO dataset annotations and images
coco_root = './train2017'

# Load COCO dataset
coco_dataset = CocoDetection(root=coco_root, annFile='./annotations/captions_train2017.json', transform=transform)

# Create DataLoader
batch_size = 1
shuffle = True
num_workers = 1
coco_loader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# initialize vae model
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

def mseloss(target, output, mode='mse'):
    # Compute squared differences
    if mode == 'mse':
        squared_diffs = (output - target) ** 2
    elif mode == 'mse_norm':
        squared_diffs = ((output - target)/output) ** 2

    # Reduce the MSE along dimension 1
    return torch.mean(squared_diffs, dim=1)


# initial mse loss
mse_loss_func = mseloss

# Example usage:
i = 0
for images, targets in coco_loader:
    original_size = images.shape[-2:]
    if original_size[0] != original_size[1]:
        square_images, cropped_size = center_crop(images)

    target_sizes = [0, 180, 210, 312, 420, 512]
    fig, axes = plt.subplots(nrows=2, ncols=1 + len(target_sizes), figsize=(5 * (1 + len(target_sizes)), 10))
    axes[0, 0].imshow(square_images.squeeze(0).permute(1, 2, 0))
    axes[0, 0].set_title(f'Original image')
    for idx, size in enumerate(target_sizes):
        if size == 0:
            # apply vae compression
            latents = vae.encode(square_images).latent_dist.sample()
            image_tensor = vae.decode(latents).sample
            if image_tensor.shape[-2:] != square_images.shape[-2:]:
                transform = transforms.Resize(square_images.shape[-2:])
                image_tensor = transform(image_tensor)
            heatmap = mse_loss_func(square_images, image_tensor).detach().numpy()
            axes[0, idx + 1].imshow(image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy())
            axes[0, idx + 1].set_title(f'No compression')
            # axes[1, idx + 1].imshow(image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy())
            # normalize heatmap
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)

            axes[1, idx + 1].imshow(heatmap.squeeze(0).transpose(1, 2, 0), cmap='viridis')
            axes[1, idx + 1].set_title(f'heatmap')
        else:
            # Define the transform
            transform = transforms.Resize((size,size))

            resized_image = transform(square_images)
            latents = vae.encode(resized_image).latent_dist.sample()
            image_tensor = vae.decode(latents).sample
            reverse_transform = transforms.Resize(cropped_size)
            reversed_image = reverse_transform(image_tensor)
            heatmap = mse_loss_func(square_images, reversed_image).detach().numpy()
            axes[0, idx + 1].imshow(reversed_image.detach().squeeze(0).permute(1, 2, 0).numpy())
            axes[0, idx + 1].set_title(str(size))
            # axes[1, idx + 1].imshow(reversed_image.detach().squeeze(0).permute(1, 2, 0).numpy())
            # normalize heatmap
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
            axes[1, idx + 1].imshow(heatmap.squeeze(0).transpose(1, 2, 0), cmap='viridis')
            axes[1, idx + 1].set_title(f'heatmap')
    fig.savefig(f'./work_dir/{i}.png')
    i += 1
    if i == 200:
        break
