import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from torchvision.models import resnet18, resnet50

from Heatmap_gen.dataloader import HeatmapDataset, RealFakeDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import entropy
from matplotlib.colors import ListedColormap

from Heatmap_gen.loss import FocalLoss
from Heatmap_gen.model import UNet, ImageTransformer
from Heatmap_gen.plot import make_mask_for_patches

import shap

from gradcam import GradCAM


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

def calculate_spatial_entropy(heatmap):
    # Normalize the heatmap
    heatmap = heatmap - torch.min(heatmap)  # Ensure all values are non-negative
    heatmap = heatmap / torch.sum(heatmap)  # Normalize to make the sum equal to 1

    # Flatten the heatmap to compute entropy
    heatmap_flat = heatmap.view(-1)
    
    # Calculate entropy
    spatial_entropy = -torch.sum(heatmap_flat * torch.log(heatmap_flat + 1e-12))  # Add a small value to avoid log(0)
    
    return spatial_entropy.item()

# with open('./real_and_compressed_dataset_coco.p', 'rb') as f:
#     data_dict = pickle.load(f)
#
# heatmap_dataset = HeatmapDataset(data_dict)

data_path = './datasets/real_fake'
heatmap_dataset = RealFakeDataset(data_path, 'train')

train_size = int(0.9 * len(heatmap_dataset))
test_size = len(heatmap_dataset) - train_size

train_dataset, test_dataset = random_split(heatmap_dataset, [train_size, test_size])
train_heatmap_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_heatmap_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = 'cuda:7'
task_name = 'real_fake_classifier_gradcam_fuse_coco_full_nopretrain_entropy_loss_clamp'
MODEL_SAVE_PATH = os.path.join('./workdir', task_name)
model_name = 'ResNet'

model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
# model = ImageTransformer().to(device)
test_images, test_labels = next(iter(test_heatmap_dataloader))
test_images = test_images.to(device)
test_labels = test_labels.to(device)
explainer = shap.DeepExplainer(model, test_images)
grad_cam1 = GradCAM(model, 'layer4.1')
grad_cam2 = GradCAM(model, 'layer3.1')
grad_cam3 = GradCAM(model, 'layer2.1')
grad_cam4 = GradCAM(model, 'layer1.1')


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
epochs = 50
fig_path = os.path.join('./workdir', task_name, 'figs')
os.makedirs(fig_path, exist_ok=True)
best_accuracy = 0
losses = np.full(epochs, np.inf)

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    correct_num = 0
    all_num = 0
    train_dataset.dataset.set_mode('train')

    for data, targets in tqdm(train_heatmap_dataloader):
        data = data.to(device)
        heatmaps_fake = (grad_cam1(data, 1) + grad_cam2(data, 1) + grad_cam3(data, 1) + grad_cam4(data, 1)) / 4
        heatmaps_real = (grad_cam1(data, 0) + grad_cam2(data, 0) + grad_cam3(data, 0) + grad_cam4(data, 0)) / 4
        fake_entropy = calculate_spatial_entropy(heatmaps_fake)
        real_entropy = calculate_spatial_entropy(heatmaps_real)
        targets = targets.squeeze(1).to(device) if len(targets.shape) == 2 else targets.to(device)
        targets = F.one_hot(targets.long(), num_classes=2).float()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(data)  # Pass the data through the model
        outputs = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, targets) + fake_entropy + real_entropy # Compute the loss
        loss.backward()  # Compute the gradient
        optimizer.step()  # Update model parameters

        predicted_classes = torch.argmax(outputs, dim=1)
        true_classes = torch.argmax(targets, dim=1)
        correct_num += (predicted_classes == true_classes).sum().item()
        all_num += len(data)
    train_accuracy = correct_num / all_num

    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct_num = 0
    test_num = 0
    test_dataset.dataset.set_mode('test')
    for real_data, fake_data, targets in tqdm(test_heatmap_dataloader):
        data = [fake_data[idx] if target == 1 else real_data[idx] for idx, target in enumerate(targets)]
        data = torch.stack(data)
        data = data.to(device)
        targets = targets.squeeze(1).to(device) if len(targets.shape) == 2 else targets.to(device)
        outputs = model(data)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        correct_num += (outputs == targets).sum().item()
        this_loss = criterion(outputs.float(), targets).item()
        val_loss += this_loss  # Sum up batch loss
        test_num += len(data)
    average_val_loss = val_loss / test_num
    val_accuracy = correct_num / test_num
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        # shap_values = explainer.shap_values(data)
        # shap_values_for_class = shap_values[0]

        # # Normalize the SHAP values to get heatmaps
        # heatmaps = shap_values_for_class - shap_values_for_class.min(axis=(1, 2, 3), keepdims=True)
        # heatmaps /= heatmaps.max(axis=(1, 2, 3), keepdims=True)
        cam1_fake, cam_output =grad_cam1(data, 1, require_output=True)
        pred_class = torch.argmax(torch.softmax(cam_output, dim=1),dim=1)
        heatmaps_fake = (cam1_fake + grad_cam2(data, 1) + grad_cam3(data, 1) + grad_cam4(data, 1)) / 4
        heatmaps_real = (grad_cam1(data, 0) + grad_cam2(data, 0) + grad_cam3(data, 0) + grad_cam4(data, 0)) / 4

        # Convert to numpy for easier manipulation if needed
        heatmaps_fake = heatmaps_fake.detach().cpu().numpy() if isinstance(heatmaps_fake, torch.Tensor) else heatmaps_fake
        heatmaps_real = heatmaps_real.detach().cpu().numpy() if isinstance(heatmaps_real, torch.Tensor) else heatmaps_real

        fig, axes = plt.subplots(nrows=4, ncols=targets.shape[0],
                                 figsize=(5 * targets.shape[0], 20))
        fig.suptitle(f'Epoch {epoch}: val accuracy {val_accuracy}')
        for i in range(targets.shape[0]):
            axes[0, i].imshow(real_data[i].permute(1, 2, 0).detach().cpu())
            axes[0, i].set_title(f'real image of label {targets[i]}, predict {pred_class[i]}')
            axes[1, i].imshow(fake_data[i].permute(1, 2, 0).detach().cpu())
            axes[1, i].set_title(f'fake image of label {targets[i]}, predict {pred_class[i]}')
            # cmap = ListedColormap([
            #     [1, 1, 1, 0],  # Transparent (RGBA)
            #     [1, 0, 0, 1]   # Red (RGBA)
            # ])
            # axes[0, i].imshow(clip_mask[i], cmap=cmap, alpha=0.5)
            # cmap = ListedColormap([
            #     [1, 1, 1, 0],  # Transparent (RGBA)
            #     [0, 0, 1, 1]  # Red (RGBA)
            # ])
            # axes[0, i].imshow(output_mask[i], cmap=cmap, alpha=0.5)
            axes[2, i].imshow(data[i].permute(1, 2, 0).detach().cpu())
            axes[2, i].imshow(heatmaps_fake[i].detach().cpu() if isinstance(heatmaps_fake, torch.Tensor) else heatmaps_fake[i], alpha=0.5)
            axes[2, i].set_title(f'target heatmap for fake class')
            axes[3, i].imshow(data[i].permute(1, 2, 0).detach().cpu())
            axes[3, i].imshow(heatmaps_real[i].detach().cpu() if isinstance(heatmaps_real, torch.Tensor) else heatmaps_real[i], alpha=0.5)
            axes[3, i].set_title(f'target heatmap for real class')
        plt.savefig(os.path.join(fig_path, f'{epoch}.png'))
        plt.close()
        print(f'best accuracy improved to {best_accuracy}')

        losses[epoch] = val_loss
        save_checkpoint(epoch, model.state_dict(), losses, MODEL_SAVE_PATH, model_name + f'_epoch_{epoch}')
    else:
        losses[epoch] = val_loss
    print(f'Epoch: {epoch}, Trian Loss: {loss.item():.4f}, Train accuracy: {train_accuracy}, Validation Loss: {val_loss:.4f}, Validation accuracy: {val_accuracy}')
