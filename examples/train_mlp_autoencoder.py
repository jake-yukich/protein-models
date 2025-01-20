import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loading import ProteinDistanceDataset, get_dataloaders, DistanceMatrixTransforms
from utils.transforms import NoiseTransforms
from models.architectures import ImprovedMLPAutoencoder
from utils.training import Trainer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys
from skimage.metrics import structural_similarity as ssim

def main():
    data_file = Path("data/test_distance_matrices_16.npy")
    if not data_file.exists():
        raise FileNotFoundError(f"Test matrices file not found: {data_file}")

    n_samples = 1000
    noise_std = 0.2
    batch_size = 32
    epochs = 50
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dir = Path("data/train_denoising")
    val_dir = Path("data/val_denoising")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original matrices...")
    original_matrices = np.load(data_file)
    matrix_dim = original_matrices.shape[1]
    print(f"Matrix dimensions: {matrix_dim}x{matrix_dim}")
    print(f"Number of original matrices: {len(original_matrices)}")

    print("\nGenerating training samples...")
    train_noisy = []
    train_clean = []
    for i in tqdm(range(n_samples), desc="Creating training data"):
        matrix_idx = np.random.randint(len(original_matrices))
        original_matrix = original_matrices[matrix_idx]
        tensor = torch.from_numpy(original_matrix).float()
        noisy_tensor = NoiseTransforms.add_gaussian_noise(tensor, std=noise_std)
        train_noisy.append(noisy_tensor.numpy())
        train_clean.append(original_matrix)
    
    np.save(train_dir / "noisy.npy", np.stack(train_noisy))
    np.save(train_dir / "clean.npy", np.stack(train_clean))

    print("\nGenerating validation samples...")
    val_noisy = []
    val_clean = []
    for i in tqdm(range(n_samples // 5), desc="Creating validation data"):
        matrix_idx = np.random.randint(len(original_matrices))
        original_matrix = original_matrices[matrix_idx]
        tensor = torch.from_numpy(original_matrix).float()
        noisy_tensor = NoiseTransforms.add_gaussian_noise(tensor, std=noise_std)
        val_noisy.append(noisy_tensor.numpy())
        val_clean.append(original_matrix)
    
    np.save(val_dir / "noisy.npy", np.stack(val_noisy))
    np.save(val_dir / "clean.npy", np.stack(val_clean))

    print("\nSetting up data loaders...")
    train_dataset = ProteinDistanceDataset(
        train_dir / "noisy.npy",
        transform=DistanceMatrixTransforms.normalize,
        return_pairs=True
    )

    val_dataset = ProteinDistanceDataset(
        val_dir / "noisy.npy",
        transform=DistanceMatrixTransforms.normalize,
        return_pairs=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("\nCreating model...")
    input_dim = matrix_dim * matrix_dim
    model = ImprovedMLPAutoencoder(
        input_dim=input_dim,
        latent_dim=512,
        hidden_dims=[2048, 1024],
        dropout=0.1,
        l1_weight=1e-5
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model device: {next(model.parameters()).device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        experiment_name="DenoisingMLP",
        device=device,
        experiment_tracker_kwargs={
            'checkpoint_frequency': 5,
            'keep_last_n': 2
        }
    )

    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': batch_size,
        'noise_std': noise_std,
        'n_samples': n_samples,
        'hidden_dims': [2048, 1024],
        'input_dim': input_dim,
        'latent_dim': 512,
        'dropout': 0.1,
        'l1_weight': 1e-5
    }

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        hyperparams=hyperparams
    )

    model.eval()
    with torch.no_grad():
        noisy_batch, clean_batch = next(iter(val_loader))
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)

        predictions = model(noisy_batch.view(noisy_batch.size(0), -1))
        predictions = predictions.view(clean_batch.size())
        
        fig_dir = train_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        for i in range(min(5, len(predictions))):
            clean = clean_batch[i].cpu().numpy()
            noisy = noisy_batch[i].cpu().numpy()
            pred = predictions[i].cpu().numpy()
            
            ssim_noisy = ssim(clean, noisy, data_range=clean.max() - clean.min())
            ssim_pred = ssim(clean, pred, data_range=clean.max() - clean.min())
            
            result_dict = {
                'noisy': noisy,
                'clean': clean,
                'predicted': pred,
                'ssim_noisy': ssim_noisy,
                'ssim_pred': ssim_pred
            }
            np.save(train_dir / f"reconstruction_{i}.npy", result_dict)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            im0 = axes[0].imshow(clean)
            axes[0].set_title('Original')
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(noisy)
            axes[1].set_title(f'Noisy (std={noise_std})\nSSIM: {ssim_noisy:.3f}')
            plt.colorbar(im1, ax=axes[1])
            
            im2 = axes[2].imshow(pred)
            axes[2].set_title(f'Reconstructed\nSSIM: {ssim_pred:.3f}')
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(fig_dir / f"reconstruction_{i}.png")
            plt.close()

    def visualize_latent_space(model, data_loader, save_dir, device, n_samples=200):
        """Visualize latent space using PCA and t-SNE"""
        print("Collecting latent vectors...")
        model.eval()
        latent_vectors = []
        original_data = []
        
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(tqdm(data_loader, desc="Processing batches")):
                if i * noisy.size(0) >= n_samples:
                    break
                
                noisy = noisy.to(device)
                latent = model.encode(noisy)
                
                latent_vectors.append(latent.cpu().numpy())
                original_data.append(clean.numpy())
        
        print("Generating visualizations...")
        latent_vectors = np.concatenate(latent_vectors, axis=0)[:n_samples]
        original_data = np.concatenate(original_data, axis=0)[:n_samples]
        
        pca = PCA(n_components=2)
        latent_2d_pca = pca.fit_transform(latent_vectors)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], 
                             c=np.mean(original_data, axis=(1,2)),  # Color by average distance
                             cmap='viridis')
        plt.colorbar(scatter, label='Average Distance')
        plt.title('PCA of Latent Space')
        plt.xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.3f})')
        plt.tight_layout()
        plt.savefig(save_dir / 'latent_space_pca.png')
        plt.close()
        
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d_tsne = tsne.fit_transform(latent_vectors)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1],
                             c=np.mean(original_data, axis=(1,2)),
                             cmap='viridis')
        plt.colorbar(scatter, label='Average Distance')
        plt.title('t-SNE of Latent Space')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(save_dir / 'latent_space_tsne.png')
        plt.close()
        
        np.savez(save_dir / 'latent_analysis.npz',
                 latent_vectors=latent_vectors,
                 pca_coords=latent_2d_pca,
                 tsne_coords=latent_2d_tsne,
                 original_data=original_data)

        print(f"Saved visualizations to {save_dir}")

    print("\nStarting visualization...")
    visualize_latent_space(
        model=model,
        data_loader=val_loader,
        save_dir=train_dir / "figures",
        device=device,
        n_samples=200
    )

    print("\nDone! Results saved in:", train_dir)

if __name__ == '__main__':
    main()