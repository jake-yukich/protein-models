import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

class ProteinDistanceDataset(Dataset):
    """Dataset for protein pairwise distance matrices"""
    def __init__(
        self,
        data_path: str | Path,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        return_pairs: bool = False
    ):
        """
        Args:
            data_path: Path to input data file (.npy)
            transform: Optional transform for input tensor
            target_transform: Optional transform for target tensor
            return_pairs: If True, will look for corresponding clean data
        """
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise ValueError(f"Path does not exist: {self.data_path}")
            
        self.transform = transform
        self.target_transform = target_transform
        self.return_pairs = return_pairs
        
        self.data = np.load(self.data_path)
        
        if return_pairs:
            clean_path = self.data_path.parent / "clean.npy"
            if not clean_path.exists():
                raise ValueError(f"Clean data not found at {clean_path}")
            self.target_data = np.load(clean_path)
        else:
            self.target_data = None
            
        if len(self.data.shape) != 3:
            raise ValueError(f"Expected 3D array (n_samples, dim, dim), got shape {self.data.shape}")
            
        self.n_samples, self.dim, _ = self.data.shape
        self.logger.info(f"Loaded {self.n_samples} matrices of size {self.dim}x{self.dim}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Load and return a distance matrix or pair of matrices
        
        Returns:
            If return_pairs=False: Single tensor of shape (dim, dim)
            If return_pairs=True: Tuple of tensors (input, target) each of shape (dim, dim)
        """
        x = torch.from_numpy(self.data[idx]).float()
        if self.transform:
            x = self.transform(x)
            
        if self.return_pairs:
            target = torch.from_numpy(self.target_data[idx]).float()
            if self.target_transform:
                target = self.target_transform(target)
            return x, target
            
        return x

def get_dataloaders(
    train_path: str | Path,
    val_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    train_dataset = ProteinDistanceDataset(train_path, **dataset_kwargs)
    val_dataset = ProteinDistanceDataset(val_path, **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

class DistanceMatrixTransforms:
    """Collection of transforms for distance matrices"""
    
    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize distances to [0,1] range"""
        return tensor / tensor.max()
    
    @staticmethod
    def add_channel_dim(tensor: torch.Tensor) -> torch.Tensor:
        """Add channel dimension for CNN input"""
        return tensor.unsqueeze(0)  # Shape: (1, N, N)
    
    @staticmethod
    def scale(tensor: torch.Tensor, factor: float) -> torch.Tensor:
        """Scale distances by a factor
        
        Args:
            tensor: Input tensor
            factor: Scaling factor to multiply values by
        """
        return tensor * factor