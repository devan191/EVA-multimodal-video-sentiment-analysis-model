import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys
import pandas as pd
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='/content/dataset1/train')
    parser.add_argument('--val-dir', type=str, default='/content/dataset1/dev')
    parser.add_argument('--test-dir', type=str, default='/content/dataset1/test')
    parser.add_argument('--model-dir', type=str, default='./saved_models')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    return parser.parse_args()

def restrict_to_n_samples(csv_path, out_path, n=5):
    df = pd.read_csv(csv_path)
    df.iloc[:n].to_csv(out_path, index=False)

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    # Restrict to 5 samples for train, dev, test
    restrict_to_n_samples(os.path.join(args.train_dir, 'train_sent_emo.csv'), 'tiny_train.csv', n=5)
    restrict_to_n_samples(os.path.join(args.val_dir, 'dev_sent_emo.csv'), 'tiny_dev.csv', n=5)
    restrict_to_n_samples(os.path.join(args.test_dir, 'test_sent_emo.csv'), 'tiny_test.csv', n=5)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv='tiny_train.csv',
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv='tiny_dev.csv',
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv='tiny_test.csv',
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    for epoch in tqdm(range(args.epochs), desc='Epochs'):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)
        print(f"Epoch {epoch+1}: train loss {train_loss['total']:.4f}, val loss {val_loss['total']:.4f}")
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase='test')
    print(f"Test loss: {test_loss['total']:.4f}")
    print(f"Test metrics: {test_metrics}")
if __name__ == '__main__':
    main()
