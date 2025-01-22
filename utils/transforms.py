import torch
import numpy as np

class NoiseTransforms:
    """Noise transformations for distance matrices"""
    
    @staticmethod
    def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.2) -> torch.Tensor:
        """Add Gaussian noise to tensor"""
        noise = torch.randn_like(tensor) * std
        noisy_tensor = tensor + noise
        return noisy_tensor.clamp(min=tensor.min().item(), max=tensor.max().item())
    
    @staticmethod
    def mask_random_entries(tensor: torch.Tensor, mask_prob: float = 0.15) -> torch.Tensor:
        """Randomly mask (set to 0) entries in the matrix"""
        mask = torch.rand_like(tensor) > mask_prob
        return tensor * mask
    
    @staticmethod
    def mask_square_region(tensor: torch.Tensor, region_size: int, random_position: bool = True) -> torch.Tensor:
        """Mask a square region in the distance matrix by setting values to 0
        
        Args:
            tensor: Input distance matrix tensor
            region_size: Size of the square region to mask
            random_position: If True, mask a random region. If False, mask center region
            
        Returns:
            Tensor with masked square region
        """
        masked = tensor.clone()
        
        if region_size > tensor.shape[0]:
            raise ValueError(f"Region size {region_size} larger than matrix dim {tensor.shape[0]}")
            
        if random_position:
            # Calculate valid starting positions
            max_start = tensor.shape[0] - region_size
            start_row = torch.randint(0, max_start + 1, (1,)).item()
            start_col = torch.randint(0, max_start + 1, (1,)).item()
        else:
            # Center the region
            start_row = (tensor.shape[0] - region_size) // 2
            start_col = start_row
            
        masked[start_row:start_row + region_size, 
                start_col:start_col + region_size] = 0
            
        return masked