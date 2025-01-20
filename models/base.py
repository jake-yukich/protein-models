import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.calculate_loss(y_pred, y)
        return loss
        
    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.calculate_loss(y_pred, y)
        return {'val_loss': loss.item()}
        
    def calculate_loss(self, y_pred, y):
        # Default MSE loss, can override
        return nn.MSELoss()(y_pred, y)

def train_model(model, train_loader, val_loader, epochs, device):
    """Basic training loop template"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            loss = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_metrics = []
            for batch in val_loader:
                val_metrics.append(model.validation_step(batch)) 