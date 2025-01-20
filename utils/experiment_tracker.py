import json
import logging
import os
from datetime import datetime
from pathlib import Path
import torch
from typing import Dict, Any

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        checkpoint_frequency: int = 5,  # Save every N epochs
        keep_last_n: int = 3  # Keep only last N checkpoints
    ):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        
        self.setup_logging()
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_metrics': []
        }
        
        self.logger.info(f"Created experiment directory: {self.experiment_dir}")
        
    def setup_logging(self):
        """Setup logging to both file and console"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        fh = logging.FileHandler(self.experiment_dir / "experiment.log")
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert values to JSON serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        return obj
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Save hyperparameters to a JSON file"""
        hyperparams = self._convert_to_serializable(hyperparams)
        
        hp_file = self.experiment_dir / "hyperparameters.json"
        with open(hp_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        self.logger.info(f"Saved hyperparameters: {hyperparams}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for current step"""
        metrics = self._convert_to_serializable(metrics)
        
        self.metrics_history['epoch_metrics'].append({
            'step': step,
            **metrics
        })
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")
        
        self._save_metrics()
    
    def save_model(self, model: torch.nn.Module, epoch: int, is_best: bool = False):
        """Save model checkpoint based on conditions"""
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        should_checkpoint = (
            epoch == 0 or
            is_best or
            (self.checkpoint_frequency > 0 and epoch % self.checkpoint_frequency == 0)
        )
        
        if should_checkpoint:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics_history': self.metrics_history
            }, checkpoint_path)
            
            self.logger.info(f"Saved model checkpoint for epoch {epoch}")
            
            if self.keep_last_n > 0:
                self._cleanup_old_checkpoints(checkpoint_dir)
        
        if is_best:
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics_history': self.metrics_history
            }, best_model_path)
            self.logger.info("Saved new best model")
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Keep only the N most recent checkpoints"""
        checkpoints = sorted([
            f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")
        ], key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
        
        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            self.logger.debug(f"Removed old checkpoint: {oldest}")
    
    def _save_metrics(self):
        """Save metrics history to JSON file"""
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4) 