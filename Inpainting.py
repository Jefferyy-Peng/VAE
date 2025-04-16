import os.path
import random

import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection
from Heatmap_gen.dataloader import DiffusionDBDataset, ArtifactDataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from torchvision.models import resnet18, resnet50
from datasets import load_dataset
import torch.nn as nn
from Heatmap_gen.model import UNet
import cv2
import matplotlib.patches as patches
from gradcam import GradCAM
from utils import Center_Crop, closest_odd, initialize_vlm
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid



def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def find_max_intensity_area(heatmap, window_size):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    if len(heatmap.shape) >= 3:
        heatmap = heatmap.squeeze(0)
    heatmap_blur = cv2.GaussianBlur(heatmap, (window_size, window_size), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_blur)
    return max_loc, max_val

def draw_box_on_heatmap(heatmap, max_pos, window_size):
    fig, ax = plt.subplots(1)
    ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    
    # Draw a rectangle
    rect = patches.Rectangle((max_pos[0] - window_size // 2, max_pos[1] - window_size // 2), window_size, window_size, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    
    plt.show()

set_seed(42)

# Define transforms for preprocessing
# transform = transforms.Compose([
#     transforms.Lambda(lambda img: img.convert('RGB')),
#     # transforms.Resize((256, 256)),  # Resize images to (256, 256)
#     transforms.ToTensor(),           # Convert images to tensors
# ])

# Path to COCO dataset annotations and images
coco_root = './train2017'

task_name = 'inpaint_test_artifact_stable_diffusion'
fig_path = os.path.join('./workdir', task_name)
os.makedirs(fig_path, exist_ok=True)


# coco_dataset = CocoDetection(root=coco_root, annFile='./annotations/captions_train2017.json', transform=transform)
# coco_loader = DataLoader(coco_dataset, batch_size=1, shuffle=False, num_workers=0)
device = 'cuda:7'

# Initialize the model
# model = UNet(3)
model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)
grad_cam1 = GradCAM(model, 'layer4.1')
grad_cam2 = GradCAM(model, 'layer3.1')
grad_cam3 = GradCAM(model, 'layer2.1')
grad_cam4 = GradCAM(model, 'layer1.1')

# load_state = torch.load('./workdir/UNet_eval/UNet.pth')
load_state = torch.load('./workdir/real_fake_classifier_gradcam_fuse_coco_full_nopretrain_entropy_loss_clamp/ResNet_epoch_9.pth')

# Load the state dictionary
model.load_state_dict(load_state['model_state_dict'])

# If you are using a GPU, you should also ensure to load the model accordingly
transform = transforms.Compose([
    Center_Crop(),
    transforms.ToTensor(),  # Convert PIL image to tensor
])

def preprocess(example):
    # Open the image using PIL and convert to RGB (if necessary)
    image = example['image'].convert('RGB')
    # Convert the PIL image to a tensor
    example['image'] = transform(image)
    return example


# dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')
# train_dataset = DiffusionDBDataset(dataset['train'], transform=transform)
train_dataset = ArtifactDataset('./datasets/artifact/stable_diffusion', transform)
dataloader = DataLoader(train_dataset, batch_size=1)
#
# import torch

# for image, prompt, seed, step, cfg, sampler, width, height, user_name, timestamp, image_nsfw, prompt_nsfw in dataloader:
i = 0
model.eval()
for data in tqdm(dataloader):
    image = data.to(device)
    # image = data['image'].to(device)
    # prompt = data_dict['prompt']
    cam1, logits = grad_cam1(image, 1, require_output=True)
    heatmaps_fake = (cam1 + grad_cam2(image, 1) + grad_cam3(image, 1) + grad_cam4(image, 1)) / 4
    predict = torch.argmax(torch.softmax(logits, dim=1))
    window_size = closest_odd(int(image.shape[-1] // 5))  # guassian blur need odd window size
    max_pos, max_val = find_max_intensity_area(heatmaps_fake, window_size)

    fig, axes = plt.subplots(nrows=3, ncols=1,
                             figsize=(5, 20))
    fig.suptitle(f'predict: {predict}')
    image = image.squeeze(0)
    axes[0].imshow(image.permute(1, 2, 0).detach().cpu())
    axes[0].set_title(f'synthesised image')
    rect = patches.Rectangle((max_pos[0] - window_size // 2, max_pos[1] - window_size // 2), window_size, window_size, linewidth=2, edgecolor='blue', facecolor='none')
    axes[0].add_patch(rect)

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
    axes[1].imshow(image.permute(1, 2, 0).detach().cpu())
    axes[1].imshow(
        heatmaps_fake.squeeze(0).detach().cpu() if isinstance(heatmaps_fake, torch.Tensor) else heatmaps_fake[i], alpha=0.5)
    rect = patches.Rectangle((max_pos[0] - window_size // 2, max_pos[1] - window_size // 2), window_size, window_size, linewidth=2, edgecolor='blue', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title(f'heatmap for fake class')
    axes[2].imshow(
        heatmaps_fake.squeeze(0).detach().cpu() if isinstance(heatmaps_fake, torch.Tensor) else heatmaps_fake[i],
        alpha=0.5)
    rect = patches.Rectangle((max_pos[0] - window_size // 2, max_pos[1] - window_size // 2), window_size, window_size, linewidth=2, edgecolor='blue', facecolor='none')
    axes[2].add_patch(rect)
    axes[2].set_title(f'heatmap for fake class')
    plt.savefig(os.path.join(fig_path, f'{i}.png'))
    plt.close()

    i += 1

# initialize vlm
captioning_model, captioning_precessor = initialize_vlm("llava-hf/llava-1.5-7b-hf", device)
prompt = "USER: <image>\nDescribe the object within ten words. If the image doesn't include an obvious object but a fraction of it, output WRONG ASSISTANT:"


pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()


# for images, targets, id in tqdm(coco_loader):

# # load base and mask image
# init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
# mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")
#
# generator = torch.Generator("cuda").manual_seed(92)
# prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
# image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
# make_image_grid([init_image, mask_image, image], rows=1, cols=3)