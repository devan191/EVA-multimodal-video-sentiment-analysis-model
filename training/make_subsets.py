import os
import pandas as pd
import shutil
from math import ceil

def make_train_subsets(base, n_splits=10):
    train_csv = os.path.join(base, 'train', 'train_sent_emo.csv')
    train_video_dir = os.path.join(base, 'train', 'train_splits')
    df = pd.read_csv(train_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    split_size = ceil(len(df) / n_splits)
    for i in range(n_splits):
        start = i * split_size
        end = min((i + 1) * split_size, len(df))
        subset_df = df.iloc[start:end]
        out_dir = os.path.join(base, f'train{i+1}')
        os.makedirs(out_dir, exist_ok=True)
        # Save CSV
        csv_out = os.path.join(out_dir, 'train_sent_emo.csv')
        subset_df.to_csv(csv_out, index=False)
        # Copy video files
        video_dst_dir = os.path.join(out_dir, 'train_splits')
        os.makedirs(video_dst_dir, exist_ok=True)
        for _, row in subset_df.iterrows():
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            src = os.path.join(train_video_dir, video_filename)
            dst = os.path.join(video_dst_dir, video_filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"[WARNING] Video not found: {src}")
        print(f"Created {csv_out} with {len(subset_df)} samples.")
    print(f"Created {n_splits} train subsets.")

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
    make_train_subsets(base, n_splits=10)

if __name__ == "__main__":
    main()
