import gradio as gr
import numpy as np
import torch
import random
from PIL import Image
from diffusers import AutoencoderKL
from pytorch_lightning import seed_everything
from torchvision import transforms

# from src.model import ClassificationModel
import hpsv2

# model = ClassificationModel.load_from_checkpoint('/mnt/storage/gait-0/xin/dev/vit-finetune/output/default/version_0/checkpoints/last.ckpt')
# model = ClassificationModel.load_from_checkpoint('/mnt/storage/gait-0/xin/dev/vit-finetune/output/384_100k_new_loss/version_0/checkpoints/best-step-step=7500-val_loss=0.0753.ckpt')

# model.eval()
# model = model.cuda()

# Load the VAE model
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

def center_crop_and_resize(image, size=384):
    """
    Crop the largest center square of the image and resize it to the specified size.
    
    Args:
    - image (numpy.ndarray): The input image as a NumPy array.
    - size (int, optional): The size to which the image is resized. Defaults to 224.
    
    Returns:
    - numpy.ndarray: The cropped and resized image as a NumPy array.
    """
    # Step 1: Determine the size of the largest square
    height, width = image.shape[:2]
    new_edge = min(height, width)
    
    # Step 2: Calculate the coordinates for cropping
    left = (width - new_edge) // 2
    top = (height - new_edge) // 2
    right = (width + new_edge) // 2
    bottom = (height + new_edge) // 2
    
    # Step 3: Crop the image
    image_cropped = image[top:bottom, left:right]
    
    # Step 4: Resize the cropped image
    image_resized = np.array(Image.fromarray(image_cropped).resize((size, size), Image.LANCZOS))

    result = Image.fromarray(image_resized)
    
    return result

def process(input_image, size, prompt = None):

    # Load your image (adjust the path to your image file)
    # image_path = "path/to/your/image.jpg"
    # image = Image.open(image_path).convert("RGB")

    # # Resize and normalize the image to match the input expected by the model
    # # Stable Diffusion VAE typically expects 256x256 images
    size = int(size)
    image = center_crop_and_resize(input_image, size)
    image = torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0  # Convert to tensor and normalize
    image = image.unsqueeze(0)  # Add batch dimension

    # Encode the image using the VAE (this gives you the latent representation)
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()

        # latents = latents.to(vae.device)
        # Decode the latents to an image tensor
        image_tensor = vae.decode(latents).sample

    # Convert the tensor to a PIL image for visualization/display
    image_tensor = image_tensor.mul(255).clamp(0, 255).byte()
    image_tensor = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = Image.fromarray(image_tensor[0])

    # input_image = center_crop_and_resize(input_image)
    score = 0
    if prompt and len(prompt) > 0:
        score = hpsv2.score(image, prompt, hps_version="v2.1") 

    # device = 'cuda'
    # # import ipdb; ipdb.set_trace()
    # input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device='cuda')
    # with torch.no_grad():
    #     score = model(input_image)
    return [image], score

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("Image quality test HPSv2")
    with gr.Row():
        input_image_0 = gr.Image(sources='upload', type="numpy")
        with gr.Column():
            size = gr.Textbox(label="Size")
            prompt = gr.Textbox(label="prompt")
    with gr.Row():
        score = gr.Textbox(label="score", value='')
        result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")

    with gr.Row():
        run_button = gr.Button()

    # with gr.Row():
    #     with gr.Column():
    #         prompt = gr.Textbox(label="Prompt")
    #         prompt_dropdown = gr.Dropdown([
    #             'anthropomorphized "Star Trek", wearing an outfit inspired by "Star Trek", detailed symmetrical face, detailed real skin textures, highly detailed, digital painting , HD quality,', 
    #             'a beautiful ukiyo painting of cyberpunk battle space pilot, wearing space techwear, detailed portrait, look at viewer, intricate complexity, concept art, by takato yamamoto, wlop, krenz cushart. cinematic dramatic atmosphere, sharp focus', 
    #             'AOT art style, ruined city background, innocent kid, tears, sunrise, angry'], 
    #             label="Avatar options")
    #         run_button = gr.Button(label="Run")
    #     with gr.Column():
    #         seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    #         control_multiplier = gr.Slider(label="Control Multiplier", minimum=0.0, maximum=5.0, value=1.5, step=0.1)

    #         with gr.Accordion("Advanced options", open=False):
    #             num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=6, step=1)
    #             image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=image_resolution_default, step=64)
    #             strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    #             guess_mode = gr.Checkbox(label='Guess Mode', value=False)
    #             detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
    #             ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
    #             scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    #             eta = gr.Number(label="eta (DDIM)", value=0.0)
    #             # a_prompt = gr.Textbox(label="Added Prompt", value='')
    #             # n_prompt = gr.Textbox(label="Negative Prompt",
    #             #                       value='')

    #             a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
    #             n_prompt = gr.Textbox(label="Negative Prompt",
    #                                   value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
    #         with gr.Accordion("Change Model", open=False):
    #             model_filename = gr.Textbox(label="Change to new model filename")
    #             change_model_button = gr.Button(value="Change")
    #             change_model_button.click(fn=change_model, inputs=model_filename, outputs=[])

    ips = [input_image_0, size, prompt]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, score])


block.launch(share=True, show_error=True)