import os
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from copy import copy
import clip
import random
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Queue

def top_crop(image):
    height, width = image.shape[-2:]
    new_edge = min(height, width)

    # Step 2: Calculate the coordinates for cropping
    left = 0
    top = 0
    right = new_edge
    bottom = new_edge

    # Step 3: Crop the image
    image_cropped = image[..., top:bottom, left:right]
    return image_cropped, image_cropped.shape[-2:]

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

def make_mask_for_patches(distance, height):
    _, indices = torch.topk(distance.view(-1), 1)
    top_indices = torch.stack((indices // height, indices % height)).t()
    mask = torch.zeros(240, 240)
    for y, x in top_indices:
        # Scale indices for the resized image
        start_x, end_x = int(x * 20), int(x * 20 + 80)
        start_y, end_y = int(y * 20), int(y * 20 + 80)
        mask[start_y:end_y, start_x] = 1  # Left vertical line
        mask[start_y:end_y, end_x - 1] = 1  # Right vertical line
        mask[start_y, start_x:end_x] = 1  # Top horizontal line
        mask[end_y - 1, start_x:end_x] = 1  # Bottom horizontal line
    return mask

def patchify_and_preprocess(image, transform=None, preprocess=None, prompt=None):
    patches = []
    input_ids = []
    masks = []
    input_id = None

    # Iterate over image with sliding window
    height = 0
    for y in range(0, 240 - 80 + 1, 20):
        width = 0
        for x in range(0, 240 - 80 + 1, 20):
            # Extract patch
            patch = image[:, :, y:y + 80, x:x + 80].squeeze(0)
            if transform is not None:
                patch = transform(transforms.ToPILImage()(patch))
            if preprocess is not None:
                if prompt is not None:
                    data = preprocess(text=prompt, images=patch, return_tensors="pt")
                    patch = data['pixel_values']
                    input_id = data['input_ids']
                    mask = data['attention_mask']
                else:
                    patch = preprocess(images=patch, return_tensors="pt").pixel_values
            patches.append(patch)
            if input_id is not None:
                input_ids.append(input_id)
                masks.append(mask)
            width += 1
        height += 1

    return patches, input_ids, masks, height, width

def find_percentile(image, low, high):
    copy_image = copy(image)
    low = np.percentile(copy_image, low)
    high = np.percentile(copy_image, high)
    copy_image[(copy_image <= low) | (copy_image >= high)] = 0
    copy_image[(copy_image > low) & (copy_image < high)] = 1
    return copy_image

def cluster_and_draw_bbox(image, eps, min_samples, ax, title):
    coordinates = np.column_stack(np.where(image > 0))
    # Apply DBSCAN to cluster coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_

    # Initialize a color image for visualization
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # Skip noise points

        # Extract coordinates of current cluster
        cluster_coords = coordinates[labels == cluster_id]

        # Find bounding box corners
        top_left_x = np.min(cluster_coords[:, 1])
        top_left_y = np.min(cluster_coords[:, 0])
        bottom_right_x = np.max(cluster_coords[:, 1])
        bottom_right_y = np.max(cluster_coords[:, 0])

        # Draw rectangle
        cv2.rectangle(color_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

        # Show the result
    ax.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)


def llava_generate(rank, world_size, model, input_ids, masks, patches, queue):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    # Move model to GPU and wrap with DDP
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Split data for this rank
    input_ids = input_ids[rank].to(rank)
    masks = masks[rank].to(rank)
    patches = patches[rank].to(rank)

    # Perform inference
    with torch.no_grad():
        generated_ids = ddp_model.module.generate(input_ids=input_ids, attention_mask=masks, pixel_values=patches,
                                                  max_length=50)

    # Move results to CPU and put in the queue
    generated_ids = generated_ids.cpu()
    queue.put(generated_ids)

    dist.destroy_process_group()


def split_data(data, num_chunks):
    """ Split data into nearly equal chunks """
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end
    return chunks

class HeatMap(nn.Module):
    def __init__(self, vae_model, mse_loss_func, lpips, clip_model, preprocess, captioning_model, captioning_precessor, plot_path=None):
        super().__init__()
        self.vae_model = vae_model
        self.mse_loss_func = mse_loss_func
        self.lpips = lpips
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.captioning_model = captioning_model
        self.captioning_precessor = captioning_precessor
        self.plot_path = plot_path

    def forward(self, square_images, kernel_radius, device, i):
        cropped_size = square_images.shape[-2:]
        prompt = "USER: <image>\nDescribe the object within ten words. If the image doesn't include an obvious object but a fraction of it, output WRONG ASSISTANT:"

        with torch.no_grad():
            latents = self.vae_model.encode(square_images).latent_dist.sample()
            image_tensor = self.vae_model.decode(latents).sample

            # orig_patches, height, width = patchify_and_preprocess(square_images.cpu(), transform=self.preprocess)
            # decoded_patches, height, width = patchify_and_preprocess(image_tensor.cpu(), transform=self.preprocess)

            ######################### commented for speed

            # scaled_square_images = (square_images * 255).to(torch.uint8)
            # scaled_image_tensor = (square_images * 255).to(torch.uint8)
            #
            # orig_patches, orig_input_ids, orig_masks, height, width = patchify_and_preprocess(scaled_square_images.cpu(), preprocess=self.captioning_precessor, prompt=prompt)
            # decoded_patches, decoded_input_ids, decoded_masks, height, width = patchify_and_preprocess(scaled_image_tensor.cpu(), transform=self.preprocess)
            #
            # # orig_patches, height, width = patchify_and_preprocess(square_images.cpu())
            # # decoded_patches, height, width = patchify_and_preprocess(image_tensor.cpu())
            #
            # # Convert list of patches to numpy array
            # orig_patches = torch.stack(orig_patches).to(device) if len(orig_patches[0].shape) == 3 else torch.stack(orig_patches).squeeze(1).to(device)
            # orig_masks = torch.stack(orig_masks).squeeze(1).to(device)
            # orig_input_ids = torch.stack(orig_input_ids).squeeze(1).to(device)
            # decoded_patches = torch.stack(decoded_patches).to(device) if len(decoded_patches[0].shape) == 3 else torch.stack(decoded_patches).squeeze(1).to(device)
            #
            # # for id, patch in enumerate(orig_patches):
            # #     np_array = patch.detach().cpu().numpy()
            # #     np_array = np.transpose(np_array, (1, 2, 0))
            # #
            # #     # Normalize the values to the range [0, 255] and convert to uint8
            # #     np_array = (np_array - np_array.min()) / (np_array.max() - np_array.min()) * 255
            # #     np_array = np_array.astype(np.uint8)
            # #
            # #     # Convert the NumPy array to a PIL image
            # #     image = Image.fromarray(np_array)
            # #     image.save(f'./patches/{id}.png')
            # # lpips_distance = lpips((orig_patches - orig_patches.min()) / (orig_patches.max() - orig_patches.min()) * 2 - 1, (decoded_patches - decoded_patches.min()) / (decoded_patches.max() - decoded_patches.min()) * 2 - 1).view(height, width)
            # # orig_features = self.clip_model.encode_image(orig_patches)
            # world_size = 32
            # split_input_ids = split_data(orig_input_ids, world_size)
            # split_masks = split_data(orig_masks, world_size)
            # split_patches = split_data(orig_patches, world_size)

            ######################### commented for speed

            # Create a multiprocessing Queue
            # queue = Queue()
            #
            # # Spawn processes
            # mp.spawn(llava_generate,
            #          args=(world_size, self.captioning_model, split_input_ids, split_masks, split_patches, queue),
            #          nprocs=world_size,
            #          join=True)
            #
            # results = []
            # while not queue.empty():
            #     results.append(queue.get())
            #
            # # Concatenate results
            # results = torch.cat(results, dim=0)
            orig_generated_ids = []
            max_length = 60

            ######################### commented for speed

            # for idx, orig_input_ids in enumerate(split_input_ids):
            #     orig_masks = split_masks[idx]
            #     orig_patches = split_patches[idx]
            #     result = self.captioning_model.module.generate(input_ids=orig_input_ids, attention_mask=orig_masks, pixel_values=orig_patches, max_length=max_length)
            #     if result.shape[1] != max_length:
            #         pad = torch.full((result.shape[0], max_length), 32001, dtype=result.dtype, device=device)
            #         pad[:, :result.shape[1]] = result
            #         result = pad
            #     orig_generated_ids.append(result)
            # orig_generated_ids = torch.cat(orig_generated_ids)
            # orig_generated_caption = [sentence[sentence.find("ASSISTANT:") + 10:].strip() for sentence in self.captioning_precessor.batch_decode(orig_generated_ids, skip_special_tokens=True)]
            # # orig_features /= orig_features.norm(dim=-1, keepdim=True)
            # orig_tokens = clip.tokenize(orig_generated_caption).to(device)
            # orig_features = self.clip_model.encode_text(orig_tokens)
            # orig_features = orig_features / orig_features.norm(dim=1, keepdim=True)
            # decoded_features = self.clip_model.encode_image(decoded_patches)
            # decoded_features = decoded_features / decoded_features.norm(dim=1, keepdim=True)
            # cosine_similarity = (torch.diagonal(orig_features @ decoded_features.T) + 1) / 2
            # clip_distance = (1 - cosine_similarity)
            # min_distance = clip_distance.min()
            # clip_distance[np.where(np.array(orig_generated_caption) == 'WRONG')] = min_distance
            # max_caption = orig_generated_caption[torch.argmax(clip_distance)]
            # clip_distance = clip_distance.view(height, width)
            # clip_mask = make_mask_for_patches(clip_distance, height)
            # # lpips_mask = make_mask_for_patches(lpips_distance, height)
            # if self.plot_path is not None:
            #     fig, axes = plt.subplots(nrows=1, ncols=3,
            #                                  figsize=(5 * 3, 5 * 1))
            #     axes[0].imshow(square_images.squeeze(0).permute(1, 2, 0).detach().cpu())
            #     axes[0].set_title(f'Original image with {str(cropped_size)}')
            #     axes[1].imshow(image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu())
            #     cmap = plt.cm.get_cmap('viridis', 2)
            #     cmap.colors[0, 3] = 0
            #     axes[1].imshow(clip_mask, cmap=cmap, alpha=0.5)
            #     axes[1].set_title(f'Reconstructed image with size 240')
            #
            #     axes[2].imshow(clip_distance.detach().cpu())
            #     axes[2].set_title(f'Caption {max_caption}')
            #
            #     fig.savefig(os.path.join(self.plot_path, f'{i}.png'))
            #     plt.close()
            #     print(f'plotted image {i}')

            ######################### commented for speed

        clip_distance = None
        return image_tensor.detach().cpu(), clip_distance


def create_mse_heatmap(square_images, model, mse_loss_func, lpips, kernel_radius, device, plot_path, i, plot=False):
    cropped_size = square_images.shape[-2:]
    target_percentile = [100]
    rows = 4

    if plot:
        fig, axes = plt.subplots(nrows=rows, ncols=1 + len(target_percentile),
                                 figsize=(5 * (1 + len(target_percentile)), 5 * rows))
        axes[0, 0].imshow(square_images.squeeze(0).permute(1, 2, 0))
        axes[0, 0].set_title(f'Original image with {str(cropped_size)}')
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    for idx, high in enumerate(target_percentile):
        # if size == 0:
        #     # apply vae compression
        #     latents = vae.encode(square_images).latent_dist.sample()
        #     image_tensor = vae.decode(latents).sample
        #     if image_tensor.shape[-2:] != square_images.shape[-2:]:
        #         transform = transforms.Resize(square_images.shape[-2:])
        #         image_tensor = transform(image_tensor)
        #     heatmap = mse_loss_func(square_images, image_tensor).detach().numpy()
        #     axes[0, idx + 1].imshow(image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy())
        #     axes[0, idx + 1].set_title(f'No compression')
        #     # axes[1, idx + 1].imshow(image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy())
        #     # normalize heatmap
        #     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
        #
        #     axes[1, idx + 1].imshow(heatmap.squeeze(0), cmap='viridis')
        #     axes[1, idx + 1].set_title(f'heatmap')
        # else:
        # Define the transform
        transform = transforms.Resize((240, 240))

        with torch.no_grad():
            resized_image = transform(square_images).to(device)
            resized_gray_img = cv2.cvtColor(resized_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2GRAY)
            latents = model.encode(resized_image).latent_dist.sample()
            image_tensor = model.decode(latents).sample.cpu()
            resized_image = resized_image.cpu()
            reverse_transform = transforms.Resize(cropped_size)
            reversed_image = reverse_transform(image_tensor)

            # encode resized image
            preprocess_resized_image = preprocess(transforms.ToPILImage()(resized_image.squeeze(0))).to(device)
            resized_features = clip_model.encode_image(preprocess_resized_image.unsqueeze(0))
            preprocess_image_tensor = preprocess(transforms.ToPILImage()(image_tensor.squeeze(0))).to(device)
            image_tensor_features = clip_model.encode_image(preprocess_image_tensor.unsqueeze(0))
            resized_features /= resized_features.norm(dim=-1, keepdim=True)
            image_tensor_features /= image_tensor_features.norm(dim=-1, keepdim=True)
            cosine_similarity = ((resized_features @ image_tensor_features.T).item() + 1) / 2
            image_clip_distance = 1 - cosine_similarity
            # resized_image_gpu = resized_image.to(device)
            # image_tensor_gpu = image_tensor.to(device)
            # image_lpips = lpips(resized_image_gpu, image_tensor_gpu)

            orig_patches, height, width = patchify_and_preprocess(resized_image, preprocess)
            decoded_patches, height, width = patchify_and_preprocess(image_tensor, preprocess)

            # Convert list of patches to numpy array
            orig_patches = torch.stack(orig_patches).to(device)
            decoded_patches = torch.stack(decoded_patches).to(device)

            # lpips_distance = lpips((orig_patches - orig_patches.min()) / (orig_patches.max() - orig_patches.min()) * 2 - 1, (decoded_patches - decoded_patches.min()) / (decoded_patches.max() - decoded_patches.min()) * 2 - 1).view(height, width)
            orig_features = clip_model.encode_image(orig_patches)
            orig_features /= orig_features.norm(dim=-1, keepdim=True)
            decoded_features = clip_model.encode_image(decoded_patches)
            decoded_features /= decoded_features.norm(dim=-1, keepdim=True)
            cosine_similarity = (torch.diagonal(orig_features @ decoded_features.T) + 1) / 2
            clip_distance = (1 - cosine_similarity).view(height, width)
            clip_mask = make_mask_for_patches(clip_distance, height)
            # lpips_mask = make_mask_for_patches(lpips_distance, height)

            clip_distance = (clip_distance - clip_distance.min()) / (clip_distance.max() - clip_distance.min())
            resized_clip_distance = F.interpolate(clip_distance.unsqueeze(0).unsqueeze(0), size=(240, 240), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().cpu().numpy()

            # lpips_distance = (lpips_distance - lpips_distance.min()) / (lpips_distance.max() - lpips_distance.min())
            # resized_lpips_distance = F.interpolate(lpips_distance.unsqueeze(0).unsqueeze(0), size=(240, 240),
            #                                       mode='bilinear', align_corners=False).squeeze(0).squeeze(
            #     0).detach().cpu().numpy()

            if plot:
                axes[1, 0].imshow(reversed_image.squeeze(0).permute(1, 2, 0))
                axes[1, 0].set_title(f'Reversed image with {str(cropped_size)}')

                axes[2, 0].imshow(resized_image.squeeze(0).permute(1, 2, 0))
                axes[2, 0].set_title(f'Resized image with size 240')

                axes[3, 0].imshow(image_tensor.squeeze(0).permute(1, 2, 0))
                cmap = plt.cm.get_cmap('viridis', 2)
                cmap.colors[0, 3] = 0
                axes[3, 0].imshow(clip_mask, cmap=cmap, alpha=0.5)
                axes[3, 0].set_title(f'Reconstructed image with size 240')

                # axes[4, 0].imshow(image_tensor.squeeze(0).permute(1, 2, 0))
                # cmap = plt.cm.get_cmap('viridis', 2)
                # cmap.colors[0, 3] = 0
                # axes[4, 0].imshow(lpips_mask, cmap=cmap, alpha=0.5)
                # axes[4, 0].set_title(f'Reconstructed image with size 240')

            heatmap_pre = mse_loss_func(resized_image, image_tensor).detach().squeeze(0).numpy()
            # heatmap_pre_erode = cv2.erode(heatmap_pre, kernel, iterations=1)
            # heatmap = mse_loss_func(square_images, reversed_image).detach().squeeze(0).numpy()
            # heatmap = cv2.erode(mse_loss_func(square_images, reversed_image).detach().squeeze(0).numpy(), kernel, iterations=1)
            # heatmap_percentile = find_percentile(heatmap, high - 10, high)
            heatmap_pre_percentile = find_percentile(heatmap_pre, high - 10, high)
            edges = cv2.Canny((resized_gray_img * 255).astype(np.uint8), 10, 200)
            # heatmap_pre_remove_edge = copy(heatmap_pre)
            # heatmap_pre_remove_edge[edges != 0] = 0
            # remove_edge_percentile = find_percentile(heatmap_pre_remove_edge, high - 10, high)
            # axes[0, idx + 1].imshow(heatmap_percentile)
            # axes[0, idx + 1].set_title('percentile of ' + str(high))
            if plot:
                axes[0, idx + 1].imshow(heatmap_pre_percentile)
                axes[0, idx + 1].set_title(f'pre-resize of percentile' + str(high))
                axes[3, idx + 1].imshow(resized_clip_distance)
                axes[3, idx + 1].set_title(f'clip_distance with whole image distance {image_clip_distance}')
                # axes[4, idx + 1].imshow(resized_lpips_distance)
                # axes[4, idx + 1].set_title(f'clip_distance with whole image distance {image_lpips}')
                axes[1, idx + 1].imshow(edges)
                axes[1, idx + 1].set_title(f'edges')
                axes[2, idx + 1].imshow(((edges == 0) & (heatmap_pre_percentile != 0)))
                axes[2, idx + 1].set_title(f'remove edges')
            # axes[2, idx + 1].imshow(gaussian_filter(heatmap_percentile, sigma=102)
            # axes[2, idx + 1].set_title(f'gaussian_filter of percentile' + str(high))
            # axes[1, idx + 1].imshow(gaussian_filter(heatmap_pre_percentile, sigma=10))
            # axes[1, idx + 1].set_title(f'gaussian_filter pre-resize of percentile' + str(high))
            smoothed_arr = gaussian_filter(heatmap_pre_percentile, sigma=10, radius=kernel_radius)
            smoothed_img = cv2.cvtColor(smoothed_arr, cv2.COLOR_GRAY2BGR)
            # edges = cv2.Canny(smoothed_img, 100, 200)
            # image_edges = cv2.Canny(resized_img, 100, 200)
            #
            # median_smoothed = smoothed_img.median()
            # median_edge = edges.median()
            center = np.where(smoothed_arr == smoothed_arr.max())
            top_left_x = max(center[1][0] - kernel_radius, 0) if center[1][
                                                                     0] <= 240 - kernel_radius else 240 - 2 * kernel_radius
            top_left_y = max(center[0][0] - kernel_radius, 0) if center[0][
                                                                     0] <= 240 - kernel_radius else 240 - 2 * kernel_radius
            bottom_right_x = min(center[1][0] + kernel_radius, 240) if center[1][
                                                                           0] >= kernel_radius else 2 * kernel_radius
            bottom_right_y = min(center[0][0] + kernel_radius, 240) if center[0][
                                                                           0] >= kernel_radius else 2 * kernel_radius

            # Draw rectangle
            cv2.rectangle(smoothed_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

            # if plot:
                # axes[1, idx + 1].imshow(smoothed_img)
                # axes[1, idx + 1].set_title(f'gaussian_filter pre-resize of percentile' + str(high))
                # cluster_and_draw_bbox(heatmap_percentile, axes[4, idx + 1], f'clustered box of percentile {str(high)}')
                # cluster_and_draw_bbox(heatmap_pre_percentile, 10, 10, axes[2, idx + 1],
                #                       f'clustered box pre-resize of percentile {str(high)}')

            cropped_image = image_tensor[0, :, top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(heatmap_percentile.astype(np.uint8), connectivity=8)
            # num_labels_pre, labels_pre, stats_pre, centroids_pre = cv2.connectedComponentsWithStats(heatmap_pre_percentile.astype(np.uint8), connectivity=8)
    if plot:
        fig.savefig(os.path.join(plot_path, f'{i}.png'))
        plt.close()
        print(f'plotted image {i}')
    return cropped_image, clip_distance



        # target_sizes = [240, 320, 416, 512, 640, 720]
        # fig, axes = plt.subplots(nrows=7, ncols=1 + len(target_sizes), figsize=(5 * (1 + len(target_sizes)), 35))
        # axes[0, 0].imshow(square_images.squeeze(0).permute(1, 2, 0))
        # axes[0, 0].set_title(f'Original image with {str(cropped_size)}')
        # for idx, size in enumerate(target_sizes):
        #     # if size == 0:
        #     #     # apply vae compression
        #     #     latents = vae.encode(square_images).latent_dist.sample()
        #     #     image_tensor = vae.decode(latents).sample
        #     #     if image_tensor.shape[-2:] != square_images.shape[-2:]:
        #     #         transform = transforms.Resize(square_images.shape[-2:])
        #     #         image_tensor = transform(image_tensor)
        #     #     heatmap = mse_loss_func(square_images, image_tensor).detach().numpy()
        #     #     axes[0, idx + 1].imshow(image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy())
        #     #     axes[0, idx + 1].set_title(f'No compression')
        #     #     # axes[1, idx + 1].imshow(image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy())
        #     #     # normalize heatmap
        #     #     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
        #     #
        #     #     axes[1, idx + 1].imshow(heatmap.squeeze(0), cmap='viridis')
        #     #     axes[1, idx + 1].set_title(f'heatmap')
        #     # else:
        #     # Define the transform
        #     transform = transforms.Resize((size,size))
        #
        #     with torch.no_grad():
        #         resized_image = transform(square_images).to(device)
        #         latents = vae.encode(resized_image).latent_dist.sample()
        #         image_tensor = vae.decode(latents).sample.cpu()
        #         resized_image = resized_image.cpu()
        #         reverse_transform = transforms.Resize(cropped_size)
        #         reversed_image = reverse_transform(image_tensor)
        #         heatmap_pre = mse_loss_func(resized_image, image_tensor).detach().squeeze(0).numpy()
        #         # heatmap_pre_erode = cv2.erode(heatmap_pre, kernel, iterations=1)
        #         heatmap = mse_loss_func(square_images, reversed_image).detach().squeeze(0).numpy()
        #         # heatmap = cv2.erode(mse_loss_func(square_images, reversed_image).detach().squeeze(0).numpy(), kernel, iterations=1)
        #         axes[0, idx + 1].imshow(reversed_image.detach().squeeze(0).permute(1, 2, 0).numpy())
        #         axes[0, idx + 1].set_title(str(size))
        #         axes[1, idx + 1].imshow(resized_image.detach().squeeze(0).permute(1, 2, 0).numpy())
        #         axes[1, idx + 1].set_title(f'resized')
        #         axes[2, idx + 1].imshow(image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy())
        #         axes[2, idx + 1].set_title(f'reconstructed')
        #         # axes[1, idx + 1].imshow(resized_image.detach().squeeze(0).permute(1, 2, 0).numpy())
        #         # axes[1, idx + 1].set_title('resized image size: ' + str(size))
        #         # axes[1, idx + 1].imshow(reversed_image.detach().squeeze(0).permute(1, 2, 0).numpy())
        #
        #         # axes[5, idx + 1].hist(heatmap.flatten(), bins=100, color='blue', alpha=0.7)
        #         # axes[5, idx + 1].set_title(f'heatmap distribution')
        #         #
        #         # axes[6, idx + 1].hist(heatmap_pre.flatten(), bins=100, color='blue', alpha=0.7)
        #         # axes[6, idx + 1].set_title(f'heatmap_pre distribution')
        #         # normalize heatmap
        #         heatmap_mean = np.mean(heatmap)
        #         # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
        #         heatmap = np.clip((heatmap - np.percentile(heatmap, 5)) / (np.percentile(heatmap, 75) - np.percentile(heatmap, 25)), 0, 1)
        #         axes[3, idx + 1].imshow(heatmap, cmap='gray')
        #         axes[3, idx + 1].set_title(f'heatmap with mean {str(round(heatmap_mean, 5))}')
        #         heatmap_pre_mean = np.mean(heatmap_pre)
        #         # heatmap_pre = (heatmap_pre - np.min(heatmap_pre)) / (np.max(heatmap_pre) - np.min(heatmap_pre) + 1e-9)
        #         heatmap_pre = np.clip((heatmap_pre - np.percentile(heatmap, 5)) / (np.percentile(heatmap_pre, 75) - np.percentile(heatmap_pre, 25)), 0, 1)
        #         axes[4, idx + 1].imshow(heatmap_pre, cmap='gray')
        #         axes[4, idx + 1].set_title(f'pre-resize heatmap with mean {str(round(heatmap_pre_mean, 5))}')
        #
        #
        #         # heatmap_pre_erode_mean = np.mean(heatmap_pre_erode)
        #         # heatmap_pre_erode = (heatmap_pre_erode - np.min(heatmap_pre_erode)) / (np.max(heatmap_pre_erode) - np.min(heatmap_pre_erode) + 1e-9)
        #         # axes[4, idx + 1].imshow(heatmap_pre_erode, cmap='gray')
        #         # axes[4, idx + 1].set_title(f'erode heatmap with mean {str(round(heatmap_pre_erode_mean, 5))}')
        #         # direct_resize_image = reverse_transform(resized_image)
        #         # heatmap_direct = cv2.erode(mse_loss_func(square_images, direct_resize_image).detach().squeeze(0).numpy(), kernel, iterations=1)
        #         # heatmap_direct_mean = np.mean(heatmap_direct)
        #         # heatmap_direct = (heatmap_direct - np.min(heatmap_direct)) / (np.max(heatmap_direct) - np.min(heatmap_direct) + 1e-9)
        #         # axes[4, idx + 1].imshow(heatmap_direct, cmap='gray')
        #         # axes[4, idx + 1].set_title(f'direct resize back heatmap: {str(round(heatmap_direct_mean, 5))}')
        #
        # fig.savefig(os.path.join(plot_path, f'{i}.png'))
        # i += 1
        # if i == 100:
        #     break

