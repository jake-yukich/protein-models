import torch
from typing import Dict, Optional
from .experiment_tracker import ExperimentTracker

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        experiment_name: Optional[str] = None,
        experiment_tracker_kwargs: Optional[Dict] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
        tracker_kwargs = experiment_tracker_kwargs or {}
        if experiment_name:
            self.tracker = ExperimentTracker(experiment_name, **tracker_kwargs)
        else:
            self.tracker = ExperimentTracker(model.__class__.__name__, **tracker_kwargs)
            
        self.tracker.logger.info(f"Model Architecture:\n{model}")
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        hyperparams: Optional[Dict] = None
    ):
        """Train the model"""
        if hyperparams:
            self.tracker.log_hyperparameters(hyperparams)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_metrics = self._train_epoch(train_loader, epoch)
            
            self.model.eval()
            val_metrics = self._validate(val_loader, epoch)
            
            metrics = {**train_metrics, **val_metrics}
            self.tracker.log_metrics(metrics, epoch)
            
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            self.tracker.save_model(self.model, epoch, is_best)
    
    def _train_epoch(self, train_loader, epoch) -> Dict[str, float]:
        """Run one training epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.model.training_step((inputs, targets))
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return {'train_loss': total_loss / len(train_loader)}
    
    def _validate(self, val_loader, epoch) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                metrics = self.model.validation_step((inputs, targets))
                total_loss += metrics['val_loss']
        
        return {'val_loss': total_loss / len(val_loader)}  