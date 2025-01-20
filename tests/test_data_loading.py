import sys
print(f"Python version: {sys.version}")

import pytest
import torch
import numpy as np
from pathlib import Path
from utils.data_loading import ProteinDistanceDataset, get_dataloaders, DistanceMatrixTransforms

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary test data"""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    
    for i in range(3):
        # Train matrices
        matrix = np.random.rand(10, 10)
        np.save(train_dir / f"matrix_{i}.npy", matrix)
        
        # Val matrices
        matrix = np.random.rand(10, 10)
        np.save(val_dir / f"matrix_{i}.npy", matrix)
    
    return tmp_path

@pytest.fixture
def single_matrix(tmp_path):
    """Create a single test matrix"""
    matrix = np.random.rand(10, 10)
    file_path = tmp_path / "test_matrix.npy"
    np.save(file_path, matrix)
    return file_path, matrix

class TestProteinDistanceDataset:
    def test_init_with_directory(self, temp_data_dir):
        dataset = ProteinDistanceDataset(temp_data_dir / "train")
        assert len(dataset) == 3
        assert dataset.dim == 10
        
    def test_init_with_single_file(self, single_matrix):
        file_path, _ = single_matrix
        dataset = ProteinDistanceDataset(file_path)
        assert len(dataset) == 1
        assert dataset.dim == 10
        
    def test_getitem_single_output(self, single_matrix):
        file_path, original_matrix = single_matrix
        dataset = ProteinDistanceDataset(file_path, return_pairs=False)
        tensor = dataset[0]
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (10, 10)
        np.testing.assert_array_almost_equal(tensor.numpy(), original_matrix)
        
    def test_getitem_pairs(self, single_matrix):
        file_path, _ = single_matrix
        dataset = ProteinDistanceDataset(file_path, return_pairs=True)
        input_tensor, target_tensor = dataset[0]
        
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
        assert torch.equal(input_tensor, target_tensor)
        
    def test_transform(self, single_matrix):
        file_path, _ = single_matrix
        transform = DistanceMatrixTransforms.normalize
        dataset = ProteinDistanceDataset(file_path, transform=transform)
        tensor = dataset[0]
        
        assert torch.max(tensor).item() == pytest.approx(1.0)
        
    def test_separate_transforms(self, single_matrix):
        file_path, _ = single_matrix
        input_transform = DistanceMatrixTransforms.normalize
        target_transform = lambda x: x * 2
        
        dataset = ProteinDistanceDataset(
            file_path,
            transform=input_transform,
            target_transform=target_transform,
            return_pairs=True
        )
        
        input_tensor, target_tensor = dataset[0]
        assert torch.max(input_tensor).item() == pytest.approx(1.0)
        assert torch.max(target_tensor).item() == pytest.approx(2.0)
        
    def test_invalid_path(self):
        with pytest.raises(ValueError):
            ProteinDistanceDataset("nonexistent_path")

class TestDataLoaders:
    def test_get_dataloaders(self, temp_data_dir):
        train_loader, val_loader = get_dataloaders(
            train_path=temp_data_dir / "train",
            val_path=temp_data_dir / "val",
            batch_size=2
        )
        
        assert len(train_loader) == 2  # 3 samples with batch_size=2 -> 2 batches
        assert len(val_loader) == 2
        
        # Check batch properties
        batch = next(iter(train_loader))
        if isinstance(batch, torch.Tensor):
            assert batch.shape == (2, 10, 10)
        else:  # tuple of tensors
            input_batch, target_batch = batch
            assert input_batch.shape == (2, 10, 10)
            assert target_batch.shape == (2, 10, 10)

class TestTransforms:
    def test_normalize(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        normalized = DistanceMatrixTransforms.normalize(tensor)
        assert torch.max(normalized).item() == 1.0
        assert normalized[1, 1].item() == 1.0  # Max should be 1
        
    def test_add_channel_dim(self):
        tensor = torch.rand(10, 10)
        with_channel = DistanceMatrixTransforms.add_channel_dim(tensor)
        assert with_channel.shape == (1, 10, 10)
        
    def test_scale(self):
        tensor = torch.ones(2, 2)
        scaled = DistanceMatrixTransforms.scale(tensor, 2.5)
        assert torch.all(scaled == 2.5)

if __name__ == "__main__":
    pytest.main([__file__]) 