import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer

# Set local dataset paths for default arguments
LOCAL_TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/train2'))
LOCAL_VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/dev'))
LOCAL_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/test'))
LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './saved_models'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")

    # Data directories (now local)
    parser.add_argument("--train-dir", type=str, default=LOCAL_TRAIN_DIR)
    parser.add_argument("--val-dir", type=str, default=LOCAL_VAL_DIR)
    parser.add_argument("--test-dir", type=str, default=LOCAL_TEST_DIR)
    parser.add_argument("--model-dir", type=str, default=LOCAL_MODEL_DIR)

    return parser.parse_args()


def main():
    
    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(
            args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )

    print(f"""Training CSV path: {os.path.join(
        args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video directory: {
          os.path.join(args.train_dir, 'train_splits')}""")

    model = MultimodalSentimentModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if requested
    checkpoint_path = os.path.join(args.model_dir, "checkpoint.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        # start_epoch = checkpoint.get('epoch', 0) + 1
        # print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss}")
    else:
        print("Starting training from scratch.")

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(start_epoch, args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        print("Training done for this epoch.")
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Log metrics in SageMaker format
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_precision",
                    "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy",
                    "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision",
                    "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy",
                    "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Ensure model directory exists before saving
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        # Save best model and checkpoint
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]

    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy",
                "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy",
                "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:emotion_precision",
                "Value": test_metrics["emotion_precision"]},
            {"Name": "test:sentiment_precision",
                "Value": test_metrics["sentiment_precision"]},
            {"Name": "test:emotion_f1",
                "Value": test_metrics["emotion_f1"]},
            {"Name": "test:sentiment_f1",
                "Value": test_metrics["sentiment_f1"]},
        ]
    }))

if __name__ == "__main__":
    main()