import torch
import torch.nn as nn
from .base import BaseModel
from typing import List, Optional

class MLPAutoencoder(nn.Module):
    """MLP-based autoencoder for distance matrices"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'relu'
    ):
        """
        Args:
            input_dim: Dimension of flattened input
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for encoder (decoder will be reversed)
            activation: Activation function to use ('relu' or 'leaky_relu')
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512]
            
        self.activation = getattr(nn.functional, activation)
        
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i + 1]),
                nn.BatchNorm1d(encoder_dims[i + 1])
            ])
            if i < len(encoder_dims) - 2:  # No activation
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                nn.BatchNorm1d(decoder_dims[i + 1])
            ])
            if i < len(decoder_dims) - 2:  # No activation
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
            
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        
        if len(x.shape) > 2:
            matrix_dim = int(reconstruction.size(1) ** 0.5)
            reconstruction = reconstruction.view(batch_size, matrix_dim, matrix_dim)
            
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        reconstruction = self.decoder(z)
        if len(reconstruction.shape) > 2:
            matrix_dim = int(reconstruction.size(1) ** 0.5)
            reconstruction = reconstruction.view(-1, matrix_dim, matrix_dim)
        return reconstruction
    
    def _ensure_on_device(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the same device as the model"""
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        return x
    
    def training_step(self, batch) -> torch.Tensor:
        """Compute loss for a training step"""
        x, target = batch
        x = self._ensure_on_device(x)
        target = self._ensure_on_device(target)
        
        original_shape = x.shape
        reconstruction = self(x)

        if len(original_shape) > 2:
            reconstruction = reconstruction.view(original_shape)
        
        mse_loss = nn.functional.mse_loss(reconstruction, target)
        
        return mse_loss
    
    def validation_step(self, batch) -> dict:
        """Compute validation metrics"""
        x, target = batch
        x = self._ensure_on_device(x)
        target = self._ensure_on_device(target)
        
        original_shape = x.shape
        reconstruction = self(x)
  
        if len(original_shape) > 2:
            reconstruction = reconstruction.view(original_shape)
        
        val_loss = nn.functional.mse_loss(reconstruction, target)
        return {'val_loss': val_loss}

# -------------------------------------------------------------------------------------------------

class ConvAutoencoder(BaseModel):
    """Convolutional Autoencoder for protein sequence analysis"""
    def __init__(self, input_channels, sequence_length):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# -------------------------------------------------------------------------------------------------

class ImprovedMLPAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        l1_weight: float = 1e-5
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [2048, 1024]
            
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i + 1]),
                nn.BatchNorm1d(encoder_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                nn.BatchNorm1d(decoder_dims[i + 1]) if i < len(decoder_dims) - 2 else nn.Identity(),
                nn.ReLU() if i < len(decoder_dims) - 2 else nn.Identity(),
                nn.Dropout(dropout) if i < len(decoder_dims) - 2 else nn.Identity()
            ])
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.l1_weight = l1_weight
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
            
        z = self.encode(x)
        reconstruction = self.decode(z)
        
        if len(x.shape) > 2:
            matrix_dim = int(x.size(1) ** 0.5)
            reconstruction = reconstruction.view(batch_size, matrix_dim, matrix_dim)
            
        return reconstruction
    
    def training_step(self, batch) -> torch.Tensor:
        x, target = batch
        x = self._ensure_on_device(x)
        target = self._ensure_on_device(target)
        
        original_shape = x.shape
        reconstruction = self(x)
        
        if len(original_shape) > 2:
            reconstruction = reconstruction.view(original_shape)
        
        mse_loss = nn.functional.mse_loss(reconstruction, target)
        l1_loss = self.l1_weight * torch.mean(torch.abs(self.encode(x)))
        
        return mse_loss + l1_loss
    
    def validation_step(self, batch) -> dict:
        x, target = batch
        x = self._ensure_on_device(x)
        target = self._ensure_on_device(target)
        
        original_shape = x.shape
        reconstruction = self(x)
        
        if len(original_shape) > 2:
            reconstruction = reconstruction.view(original_shape)
        
        val_loss = nn.functional.mse_loss(reconstruction, target)
        return {'val_loss': val_loss}
    
    def _ensure_on_device(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        return x

# -------------------------------------------------------------------------------------------------

class VariationalAutoencoder(BaseModel):
    """Variational Autoencoder implementation"""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        activation: str = 'relu'
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
            
        encoder_layers = []
        last_h = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_h, h_dim))
            if activation == 'relu':
                encoder_layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                encoder_layers.append(nn.LeakyReLU())
            last_h = h_dim
            
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(last_h, latent_dim)
        self.fc_var = nn.Linear(last_h, latent_dim)
        
        decoder_layers = []
        last_h = latent_dim
        hidden_dims.reverse()
        for h_dim in hidden_dims:
            decoder_layers.append(nn.Linear(last_h, h_dim))
            if activation == 'relu':
                decoder_layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                decoder_layers.append(nn.LeakyReLU())
            last_h = h_dim
        decoder_layers.append(nn.Linear(last_h, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to get mean and log variance of latent distribution"""
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        hidden = self.encoder_layers(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from latent distribution"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        reconstruction = self.decoder(z)
        if len(reconstruction.shape) > 2:
            matrix_dim = int(reconstruction.size(1) ** 0.5)
            reconstruction = reconstruction.view(-1, matrix_dim, matrix_dim)
        return reconstruction
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mean and log variance"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
        
    def training_step(self, batch) -> torch.Tensor:
        """Compute ELBO loss for training"""
        x, target = batch
        reconstruction, mu, log_var = self(x)
        
        recon_loss = nn.functional.mse_loss(reconstruction, target)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kl_loss
        
        return total_loss
        
    def validation_step(self, batch) -> dict:
        """Compute validation metrics"""
        x, target = batch
        reconstruction, mu, log_var = self(x)
        
        recon_loss = nn.functional.mse_loss(reconstruction, target)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kl_loss
        
        return {
            'val_loss': total_loss,
            'val_recon_loss': recon_loss,
            'val_kl_loss': kl_loss
        }

# -------------------------------------------------------------------------------------------------

# class SimpleMLP(BaseModel):
#     """Simple multi-layer perceptron model"""
#     def __init__(self, input_dim, hidden_dims, output_dim):
#         super().__init__()
        
#         layers = []
#         prev_dim = input_dim
        
#         for hidden_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Dropout(0.2)
#             ])
#             prev_dim = hidden_dim
            
#         layers.append(nn.Linear(prev_dim, output_dim))
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.network(x)

# class LSTMModel(BaseModel):
#     """Simple LSTM model for sequence data"""
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super().__init__()
        
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         output = self.fc(lstm_out[:, -1, :])
#         return output 
 