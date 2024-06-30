import random

import numpy as np
import torch.cuda
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from Heatmap_gen.model import UNet

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def initialize_vlm(device):
    checkpoint = "microsoft/git-base"
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    return model,  processor
set_seed(42)

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    # transforms.Resize((256, 256)),  # Resize images to (256, 256)
    transforms.ToTensor(),           # Convert images to tensors
])

# Path to COCO dataset annotations and images
coco_root = './train2017'

coco_dataset = CocoDetection(root=coco_root, annFile='./annotations/captions_train2017.json', transform=transform)
coco_loader = DataLoader(coco_dataset, batch_size=1, shuffle=False, num_workers=0)

# Initialize the model
model = UNet(3)

load_state = torch.load('./workdir/UNet_eval/UNet.pth')

# Load the state dictionary
model.load_state_dict(load_state['model_state_dict'])

# If you are using a GPU, you should also ensure to load the model accordingly
device = 'cuda:7'
model.to(device)
#
# import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

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