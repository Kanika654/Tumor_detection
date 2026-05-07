import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BraTSDataset(Dataset):
    def __init__(self, metadata_path, base_dir, split='train', img_size=128):
        self.metadata = pd.read_csv(metadata_path)
        self.base_dir = base_dir
        self.img_size = img_size
        self.split = split
        
        # Simple split logic (e.g. 80-20 partition)
        np.random.seed(42)
        idx = np.random.permutation(len(self.metadata))
        split_point = int(0.8 * len(self.metadata))
        
        if split == 'train':
            self.metadata = self.metadata.iloc[idx[:split_point]]
        else:
            self.metadata = self.metadata.iloc[idx[split_point:]]
            
        self.metadata = self.metadata.reset_index(drop=True)
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        idx = int(idx)
        
        row = self.metadata.iloc[idx]
        # Paths are stored as /content/data/...
        rel_path = row['slice_path'].lstrip('/')
        abs_path = os.path.join(self.base_dir, rel_path)
        
        # Load h5 using h5py. The typical key is 'image'.
        with h5py.File(abs_path, 'r') as f:
            # We check the first key, typically 'image' and 'mask' in this competition
            key = list(f.keys())[0] if 'image' not in f.keys() else 'image'
            img = np.array(f[key])
        
        # Post-processing to normalize to ~[-1, 1] for cGAN
        # Handle multi-channel logic (if say 4 sequences) or single channel
        if len(img.shape) == 3 and img.shape[-1] <= 4:
            # Channels last -> Channels first
            img = np.transpose(img, (2, 0, 1))
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0) # add channel dim

        tensor = torch.tensor(img, dtype=torch.float32)

        # Normalize, Augment and Resize
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), antialias=True),
            ])
        
        tensor = transform(tensor)
        
        # Normalization [0, 1] min-max then scaled to [-1, 1]
        t_max = tensor.max()
        t_min = tensor.min()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
        tensor = tensor * 2.0 - 1.0
        
        target = torch.tensor(row['target'], dtype=torch.float32)
        
        return tensor, target

class MergedDataset(Dataset):
    def __init__(self, original_dataset, synthetic_images, synthetic_targets):
        self.original_dataset = original_dataset
        self.synth_images = synthetic_images
        self.synth_targets = synthetic_targets
        
        # Add the same augmentation as the train split
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
        
    def __len__(self):
        return len(self.original_dataset) + len(self.synth_images)
        
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            synth_idx = idx - len(self.original_dataset)
            img = self.synth_images[synth_idx]
            if self.transform:
                img = self.transform(img)
            return img, self.synth_targets[synth_idx]
