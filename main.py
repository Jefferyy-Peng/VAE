import os

import clip
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
from torch import nn
from torch.nn import DataParallel
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import numpy as np
import torch
import cv2
from tqdm import tqdm
import random
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import lpips
import torch.multiprocessing as mp
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration

from torch.nn.parallel import DistributedDataParallel as DDP

from Heatmap_gen.dataloader import HumanImageDataset
from Inpainting import initialize_vlm
from VAE_l2_score import create_mse_heatmap, center_crop, top_crop, HeatMap
import pickle

class Top_Crop:
    def __init__(self):
        pass

    def __call__(self, image):
        height, width = image.shape[-2:]
        new_edge = min(height, width)

        # Step 2: Calculate the coordinates for cropping
        left = 0
        top = 0
        right = new_edge
        bottom = new_edge

        # Step 3: Crop the image
        image_cropped = image[..., top:bottom, left:right]
        return image_cropped


class Center_Crop:
    def __init__(self):
        pass

    def __call__(self, image):
        image = np.array(image)
        if image.shape[-1] == 3:
            height, width = image.shape[:2]
            new_edge = min(height, width)

            # Step 2: Calculate the coordinates for cropping
            left = (width - new_edge) // 2
            top = (height - new_edge) // 2
            right = (width + new_edge) // 2
            bottom = (height + new_edge) // 2

            # Step 3: Crop the image
            image_cropped = image[top:bottom, left:right, ...]
            image_cropped = Image.fromarray(image_cropped)
        elif image.shape[0] == 3:
            height, width = image.shape[-2:]
            new_edge = min(height, width)

            # Step 2: Calculate the coordinates for cropping
            left = (width - new_edge) // 2
            top = (height - new_edge) // 2
            right = (width + new_edge) // 2
            bottom = (height + new_edge) // 2

            # Step 3: Crop the image
            image_cropped = image[..., top:bottom, left:right]
            image_cropped = Image.fromarray(image_cropped)
        else:
            raise NotImplementedError
        return image_cropped


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def initialize_vlm(checkpoint_name, device):
    checkpoint = checkpoint_name
    if checkpoint == "microsoft/git-base":
        processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    elif checkpoint == 'llava-hf/llava-1.5-7b-hf':
        model = LlavaForConditionalGeneration.from_pretrained(checkpoint)
        processor = AutoProcessor.from_pretrained(checkpoint)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = DataParallel(model).to(device)
        else:
            model = model.to(device)

    return model,  processor

seed_value = 42
set_seed(seed_value)

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((256, 256)),  # Resize images to (256, 256)
    transforms.ToTensor(),           # Convert images to tensors
])

# Path to COCO dataset annotations and images
coco_root = './train2017'
plot_path = './semantic_degradation_human'
os.makedirs(plot_path, exist_ok=True)

# Load COCO dataset
# Create DataLoader
batch_size = 1
shuffle = False
num_workers = 0


# initialize vae model
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae.eval()


# git, git_processor = initialize_vlm(device)

def mseloss(target, output, mode='mse'):
    # Compute squared differences
    if mode == 'mse':
        squared_diffs = (output - target) ** 2
    elif mode == 'mse_norm':
        squared_diffs = ((output - target)/target) ** 2

    # Reduce the MSE along dimension 1
    return torch.mean(squared_diffs, dim=1)


# initial mse loss
mse_loss_func = mseloss

kernel = np.ones((2, 2), np.uint8)
kernel_radius = 50
to_pil = transforms.ToPILImage()
lpips = lpips.LPIPS(net='alex')
distance_map_dict = {}
clip_model, preprocess = clip.load("ViT-B/32")

def main(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    # setup(rank, world_size)

    # create model and move it to GPU with id rank
    # torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    # device = 'cuda'

    captioning_model, captioning_precessor = initialize_vlm("microsoft/git-base", device)
    # captioning_model, captioning_precessor = initialize_vlm("llava-hf/llava-1.5-7b-hf", device)
    heatmap_gen = HeatMap(vae, mse_loss_func, lpips, clip_model, preprocess, captioning_model, captioning_precessor, plot_path).to(device)
    # heatmap_gen = DDP(heatmap_gen, device_ids=[rank])
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        Center_Crop(),
        transforms.Resize((240, 240)),  # Resize images to (256, 256)
        transforms.ToTensor(),  # Convert images to tensors
    ])
    dataset = CocoDetection(root=coco_root, annFile='./annotations/captions_train2017.json', transform=transform)
    # dataset = HumanImageDataset('./datasets/SHHQ-1.0/no_segment', transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    i = 0
    data_images = []
    targets = []
    real_path = './datasets/real_fake/real'
    fake_path = './datasets/real_fake/fake'
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)

    # for images, targets, id in tqdm(coco_loader):
    for images in tqdm(data_loader):

        try: 
            # original_size = images.shape[-2:]
            # if original_size[0] != original_size[1]:
            #     square_images, cropped_size = top_crop(images)
            # else:
            #     square_images, cropped_size = images, original_size
            #
            # square_images = square_images.to(device)
            # transform = transforms.Resize((240, 240))
            # square_images = transform(square_images)
            square_images = images[0].to(device)

            # cropped_image, clip_distance = create_mse_heatmap(square_images, vae, mse_loss_func, lpips, kernel_radius, device, plot_path, i, plot=True)
            image_tensor, clip_distance = heatmap_gen(square_images, kernel_radius, device, i)
            image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0)
            data_images.append(image_tensor)
            data_images.append(square_images.cpu())
            targets.append(torch.ones(len(image_tensor)))
            targets.append(torch.zeros(len(square_images)))
            # pil_image = to_pil(cropped_image)
            # pixel_values = git_processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
            # generated_ids = git.generate(pixel_values=pixel_values, max_length=50)
            # generated_caption = git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            distance_map_dict[id] = [image_tensor, clip_distance]
            real_image = (square_images.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            real_image = Image.fromarray(real_image)
            real_image.save(os.path.join(real_path, f'{i}.png'))
            fake_image = (image_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            fake_image = Image.fromarray(fake_image)
            fake_image.save(os.path.join(fake_path, f'{i}.png'))
            i += 1
            # if i == 45000:
            #     # with open('./heatmap_semantic_degradation_human.p', 'wb') as f:
            #     #     pickle.dump(distance_map_dict, f)
                # with open('./real_and_compressed_dataset_coco.p', 'wb') as f:
                #     pickle.dump({'data': torch.cat(data_images), 'target': torch.cat(targets)}, f)
                # break
        except KeyboardInterrupt:
            # If you catch the interrupt in the main loop, you can handle it here too
            print('Interrupted! Saving data...')
            # with open('./heatmap_text_image.p', 'wb') as f:
            #     pickle.dump(distance_map_dict, f)
            break
    # # cleanup()
    # with open('./real_and_compressed_dataset_coco.p', 'wb') as f:
    #     pickle.dump({'data': torch.cat(data_images), 'target': torch.cat(targets)}, f)


def run_main(main_fn, world_size):
    mp.spawn(main_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    # n_gpus = 8
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    # world_size = n_gpus
    # run_main(main, world_size)
    main(None, 1)

