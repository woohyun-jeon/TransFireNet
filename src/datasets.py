import os
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset


class BurnSeverityDataset(Dataset):
    def __init__(self, data_dir, dataset_ids, transform=None):
        self.image_dir = os.path.join(data_dir, 'image')
        self.label_dir = os.path.join(data_dir, 'label')
        self.dataset_ids = dataset_ids
        self.transform = transform

        self.pre_mean = np.array([2645.19185232246, 2744.001138373883, 2119.2794981281254], dtype=np.float32)
        self.pre_std = np.array([934.995081474556, 989.8159677271037, 924.8957662027137], dtype=np.float32)
        self.post_mean = np.array([2245.19185232246, 2544.001138373883, 1919.2794981281254], dtype=np.float32)
        self.post_std = np.array([834.995081474556, 889.8159677271037, 824.8957662027137], dtype=np.float32)

        self.before_files = []
        self.after_files = []
        self.label_files = []

        for dataset_id in self.dataset_ids:
            before_file = os.path.join(self.image_dir, 'before', dataset_id)
            after_file = os.path.join(self.image_dir, 'after', dataset_id)
            label_file = os.path.join(self.label_dir, dataset_id)

            if os.path.exists(before_file) and os.path.exists(after_file) and os.path.exists(label_file):
                self.before_files.append(before_file)
                self.after_files.append(after_file)
                self.label_files.append(label_file)

    def __len__(self):
        return len(self.before_files)

    def __getitem__(self, idx):
        before_path = self.before_files[idx]
        after_path = self.after_files[idx]
        label_path = self.label_files[idx]

        before_img = self.load_image(before_path)
        after_img = self.load_image(after_path)
        label = self.load_label(label_path)

        augmented = self.transform(image=before_img, image2=after_img, mask=label)
        before_img = augmented['image']
        after_img = augmented['image2']
        label = augmented['mask']

        pre_mean = torch.from_numpy(self.pre_mean).view(-1, 1, 1)
        pre_std = torch.from_numpy(self.pre_std).view(-1, 1, 1)
        post_mean = torch.from_numpy(self.post_mean).view(-1, 1, 1)
        post_std = torch.from_numpy(self.post_std).view(-1, 1, 1)

        before_img = (before_img - pre_mean) / pre_std
        after_img = (after_img - post_mean) / post_std

        label = torch.as_tensor(label, dtype=torch.long)
        return before_img, after_img, label, os.path.basename(before_path)

    def load_image(self, path):
        with rasterio.open(path) as src:
            img = src.read([8, 9, 10]).astype(np.float32)
            img = np.transpose(img, (1, 2, 0))
        return img

    def load_label(self, path):
        with rasterio.open(path) as src:
            label = src.read(1).astype(np.int64)
        return label


def get_transforms():
    common_transforms = [
        A.Resize(256, 256),
    ]

    aug_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ]

    train_transform = A.Compose(
        common_transforms + aug_transforms + [ToTensorV2()],
        additional_targets={'image2': 'image'}
    )

    val_transform = A.Compose(
        common_transforms + [ToTensorV2()],
        additional_targets={'image2': 'image'}
    )

    return train_transform, val_transform


def get_siamese_datasets(data_path, train_ids, val_ids, test_ids):
    train_transform, val_transform = get_transforms()

    train_dataset = BurnSeverityDataset(data_path, train_ids, transform=train_transform)
    val_dataset = BurnSeverityDataset(data_path, val_ids, transform=val_transform)
    test_dataset = BurnSeverityDataset(data_path, test_ids, transform=val_transform)

    return train_dataset, val_dataset, test_dataset
