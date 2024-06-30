import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from Heatmap_gen.dataloader import HeatmapDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib.colors import ListedColormap

from Heatmap_gen.loss import FocalLoss
from Heatmap_gen.model import UNet, ImageTransformer
from Heatmap_gen.plot import make_mask_for_patches


def save_checkpoint(epoch, state_dict, losses, MODEL_SAVE_PATH, model_name):
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": state_dict,
        "losses": losses,
        "best_loss": min(losses),
    }, os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth"))


def min_max_normalize_2d(data):
    """
    Normalize each feature in a 2D array using the Min-Max scaling technique.

    Parameters:
        data (numpy.ndarray): The 2D data to normalize, where each column represents a feature.

    Returns:
        numpy.ndarray: Min-Max normalized data.
    """
    min_val = torch.min(data.view(data.shape[0], -1), dim=1).values  # Minimum value for each feature
    max_val = torch.max(data.view(data.shape[0], -1), dim=1).values  # Maximum value for each feature

    # Avoid division by zero for constant features
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    normalized_data = (data - min_val[:, None, None]) / range_val[:, None, None]
    return normalized_data

with open('./heatmap.p', 'rb') as f:
    data = pickle.load(f)

data_dict = {'data': [], 'target': []}
for key, value in data.items():
    data_dict['data'].append(value[0].squeeze(0))
    data_dict['target'].append(value[1])

heatmap_dataset = HeatmapDataset(data_dict)

train_size = int(0.8 * len(heatmap_dataset))
test_size = len(heatmap_dataset) - train_size

train_dataset, test_dataset = random_split(heatmap_dataset, [train_size, test_size])
train_heatmap_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_heatmap_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = 'cuda:7'
task_name = 'UNet_eval'
MODEL_SAVE_PATH = os.path.join('./workdir', task_name)
model_name = 'UNet'

model = UNet(3).to(device)
# model = ImageTransformer().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50
fig_path = os.path.join('./workdir', task_name, 'figs')
os.makedirs(fig_path, exist_ok=True)
best_loss = np.inf
losses = np.full(epochs, np.inf)

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    for data, targets in tqdm(train_heatmap_dataloader):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(data)  # Pass the data through the model
        loss = criterion(outputs, targets)  # Compute the loss
        loss.backward()  # Compute the gradient
        optimizer.step()  # Update model parameters

    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct = 0
    with torch.no_grad():  # Turn off gradients for validation
        for data, targets in tqdm(test_heatmap_dataloader):
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            outputs = min_max_normalize_2d(outputs)
            targets = min_max_normalize_2d(targets)
            output_mask = make_mask_for_patches(outputs, outputs.shape[-1])
            clip_mask = make_mask_for_patches(targets, outputs.shape[-1])
            this_loss = criterion(outputs, targets).item()
            val_loss += this_loss  # Sum up batch loss
            if this_loss < best_loss:
                best_loss = this_loss
                fig, axes = plt.subplots(nrows=3, ncols=targets.shape[0],
                                         figsize=(5 * targets.shape[0], 15))
                for i in range(targets.shape[0]):
                    axes[0, i].imshow(data[i].permute(1, 2, 0).detach().cpu())
                    axes[0, i].set_title(f'image of {i}')
                    cmap = ListedColormap([
                        [1, 1, 1, 0],  # Transparent (RGBA)
                        [1, 0, 0, 1]   # Red (RGBA)
                    ])
                    axes[0, i].imshow(clip_mask[i], cmap=cmap, alpha=0.5)
                    cmap = ListedColormap([
                        [1, 1, 1, 0],  # Transparent (RGBA)
                        [0, 0, 1, 1]  # Red (RGBA)
                    ])
                    axes[0, i].imshow(output_mask[i], cmap=cmap, alpha=0.5)
                    axes[1, i].imshow(outputs[i].detach().cpu())
                    axes[1, i].set_title(f'output heatmap of {i}')
                    axes[2, i].imshow(targets[i].detach().cpu())
                    axes[2, i].set_title(f'target heatmap of {i}')
                plt.savefig(os.path.join(fig_path, f'{epoch}.png'))
                plt.close()
                print(f'best loss improved to {best_loss}')

    val_loss /= len(test_heatmap_dataloader.dataset)
    if val_loss < losses.min():
        losses[epoch] = val_loss
        save_checkpoint(epoch, model.state_dict(), losses, MODEL_SAVE_PATH, model_name)
    else:
        losses[epoch] = val_loss
    print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')
