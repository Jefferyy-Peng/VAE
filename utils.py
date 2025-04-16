import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel

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

def closest_odd(number):
    # If the number is even, adjust it to the closest odd number
    if number % 2 == 0:
        return number + 1
    else:
        return number