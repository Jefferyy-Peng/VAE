import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image as Image
import os
import pandas as pd

class DiffusionDBDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'prompt': item['prompt'],
        }

class ArtifactDataset(Dataset):
    def __init__(self, root_path, transform):
        df = pd.read_csv(os.path.join(root_path, 'metadata.csv'))
        self.root_path = root_path
        self.path_list = df['image_path']
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = os.path.join(self.root_path, self.path_list[index])
        image = Image.open(path)
        image = self.transform(image)
        return image

class HeatmapDataset(Dataset):
    def __init__(self, data_dict, mode, transform=None):
        """
        Args:
            data_dict (dict): A dictionary with 'images' and 'masks' as keys.
                              The values should be lists of file paths.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict['data'])

    def __getitem__(self, idx):
        # Load image and mask from file paths
        image = self.data_dict['data'][idx]
        mask = self.data_dict['target'][idx]

        return image, mask.float()

class RealFakeDataset(Dataset):
    def __init__(self, root_path, mode, transform=None):
        """
        Args:
            data_dict (dict): A dictionary with 'images' and 'masks' as keys.
                              The values should be lists of file paths.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        real_data_path = os.path.join(root_path, 'real')
        fake_data_path = os.path.join(root_path, 'fake')
        real_data = [os.path.join(real_data_path, path) for path in os.listdir(real_data_path)]
        fake_data = [os.path.join(fake_data_path, path) for path in os.listdir(fake_data_path)]
        self.mode = mode
        self.target_list = [0] * len(real_data) + [1] * len(fake_data)
        self.data_list = real_data + fake_data
        self.transform = transform

    def set_mode(self, mode):
        self.mode = mode
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Load image and mask from file paths
        if self.mode == 'train':
            image_path = self.data_list[idx]
            image = Image.open(image_path)
            image = torch.Tensor(np.array(image).transpose(2, 0, 1) / 255)
            mask = self.target_list[idx]

            return image, torch.Tensor([mask]).float()
        else:
            image_path = self.data_list[idx]
            if image_path.split('/')[3] == 'fake':
                opposite_path = image_path.replace('/fake/', '/real/')
                fake_image = Image.open(image_path)
                fake_image = torch.Tensor(np.array(fake_image).transpose(2, 0, 1) / 255)
                mask = self.target_list[idx]
                real_image = Image.open(opposite_path)
                real_image = torch.Tensor(np.array(real_image).transpose(2, 0, 1) / 255)
            else:
                opposite_path = image_path.replace('/real/', '/fake/')
                real_image = Image.open(image_path)
                real_image = torch.Tensor(np.array(real_image).transpose(2, 0, 1) / 255)
                mask = self.target_list[idx]
                fake_image = Image.open(opposite_path)
                fake_image = torch.Tensor(np.array(fake_image).transpose(2, 0, 1) / 255)

            return real_image, fake_image, torch.Tensor([mask]).float()


class HumanImageDataset(Dataset):
    def __init__(self, image_directory, transform=None):
        """
        Args:
            image_directory (string): Path to the image directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_directory = image_directory
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_directory) if
                                os.path.isfile(os.path.join(image_directory, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_directory, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')  # Convert to RGB to ensure consistency

        if self.transform:
            image = self.transform(image)

        return image