import argparse
from pathlib import Path
import torch
from models import get_model
from data import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate protein structure models')
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture to use')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=Path, default='data')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=Path, default='outputs')
    return parser.parse_args()

def main():
    args = parse_args()
  
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(args.device)
    loader = get_dataloader(args.data_dir, args.split, args.batch_size)
    
    model.eval()
    metrics = {}
    with torch.no_grad():
        for batch in loader:
            batch_metrics = model.validation_step(batch)
    
    print(f"Results on {args.split} set:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main() 