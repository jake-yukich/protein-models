import argparse
from pathlib import Path
import torch
from utils.training import Trainer
from models import get_model
from data import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Train protein structure models')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model architecture to use')
    parser.add_argument('--data-dir', type=Path, default='data',
                       help='Directory containing dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=Path, default='outputs',
                       help='Directory to save checkpoints and logs')
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    train_loader = get_dataloader(args.data_dir, 'train', args.batch_size)
    val_loader = get_dataloader(args.data_dir, 'val', args.batch_size)
    
    model = get_model(args.model)
    trainer = Trainer(
        model=model,
        device=args.device,
        log_dir=args.output_dir
    )

    trainer.train(train_loader, val_loader, args.epochs)

if __name__ == '__main__':
    main() 