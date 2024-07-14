import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        self.transform = transforms.ToTensor()
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = F.one_hot(torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt")), num_classes=4)
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.transform(self.X[i]), self.y[i], self.subject_idxs[i]
        else:
            return self.transform(self.X[i]), self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

class ThingsMEGDataset_mod(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        self.transform = transforms.ToTensor()
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).unsqueeze(3)
        self.subject_idxs = F.one_hot(torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt")), num_classes=4).unsqueeze(2).unsqueeze(3)
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.transform(self.X[i]), self.y[i], self.subject_idxs[i]
        else:
            return self.transform(self.X[i]), self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

class ThingsMEGDataset_mod1d(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        self.transform = transforms.ToTensor()
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = F.one_hot(torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt")), num_classes=4).unsqueeze(2).expand(-1, -1, 281)
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.transform(self.X[i]), self.y[i], self.subject_idxs[i]
        else:
            return self.transform(self.X[i]), self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]