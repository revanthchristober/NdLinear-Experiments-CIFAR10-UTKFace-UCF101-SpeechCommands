# ndlinear_project/ucf101/dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io # Use torchvision.io for video reading

class UCF101Dataset(Dataset):
    """
    Custom Dataset for loading clips from UCF101 dataset.
    Assumes structure: root_dir/split/class/*.avi
    """
    def __init__(self, root_dir: str, split: str, frames_per_clip: int = 4, transform=None, frame_step: int = 1):
        self.root_dir = root_dir
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.frame_step = frame_step # Control step between sampled frames

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
             raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Collect class names
        self.classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        if not self.classes:
             raise FileNotFoundError(f"No class subdirectories found in {split_dir}")
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Build list of (video_path, label_idx) samples
        self.samples = []
        print(f"Loading {split} samples from {split_dir}...")
        for cls_idx, cls in enumerate(self.classes):
            class_dir = os.path.join(split_dir, cls)
            video_files = glob.glob(os.path.join(class_dir, '*.avi'))
            if not video_files:
                print(f"Warning: No .avi files found in {class_dir}")
                continue
            for video_path in video_files:
                self.samples.append((video_path, cls_idx))

        if not self.samples:
             raise RuntimeError(f"No video samples found for split '{split}' in {root_dir}")
        print(f"Found {len(self.samples)} samples for split '{split}'.")


    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, num_frames: int):
        """Helper to select frame indices"""
        if num_frames < self.frames_per_clip * self.frame_step:
            # Not enough frames, repeat last frame
            indices = np.arange(0, num_frames, self.frame_step)
            num_missing = self.frames_per_clip - len(indices)
            if num_missing > 0:
                last_idx = indices[-1] if len(indices) > 0 else num_frames - 1
                padding = np.full(num_missing, last_idx, dtype=int)
                indices = np.concatenate((indices, padding))
            return indices[:self.frames_per_clip] # Ensure correct length
        else:
            # Enough frames, sample evenly spaced
            # Select indices such that frame_step is considered
            # Example: frames=16, clip=4, step=2 -> need indices like [0, 2, 4, 6] or [1, 3, 5, 7] etc.
            # We want T indices spanning the video duration num_frames
            available_length = num_frames - (self.frames_per_clip - 1) * self.frame_step
            start_idx = np.random.randint(0, available_length) if self.split == 'train' else 0 # Random start for train
            indices = np.arange(self.frames_per_clip) * self.frame_step + start_idx
            return indices


    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        try:
            # Read video using torchvision.io
            # pts_unit='sec' helps handle variable frame rates if any
            video_frames, audio_frames, info = io.read_video(video_path, pts_unit='sec', output_format="TCHW")
            # video_frames: [T, C, H, W], uint8 tensor
        except Exception as e:
            print(f"Error reading video {video_path}: {e}. Returning dummy data.")
            # Return a dummy clip and label
            dummy_clip = torch.zeros((self.frames_per_clip, 3, 128, 128), dtype=torch.float32)
            return dummy_clip, -1 # Use -1 label to indicate error

        num_frames = video_frames.shape[0]
        indices = self._sample_frame_indices(num_frames)

        # Select frames and normalize to [0, 1] float
        clip = video_frames[indices].float() / 255.0
        # clip shape: [T_clip, C, H, W]

        # Apply transforms if any (expecting CxHxW)
        if self.transform:
            # Apply transform to each frame in the clip
            transformed_clip = torch.stack([self.transform(frame) for frame in clip], dim=0)
        else:
            transformed_clip = clip

        # Ensure output shape is [T_clip, C, H, W]
        return transformed_clip, label


def get_ucf101_dataloaders(data_root: str, frames_per_clip: int, batch_size: int, frame_step: int = 1, num_workers: int = 4):
    """Creates UCF101 DataLoaders using the custom dataset."""

    # Define transforms for each frame
    # Resize and Normalize
    transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True), # Use antialias for better quality
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats often used
        # Or use [0.5]*3, [0.5]*3 for [-1, 1] normalization
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create datasets
    train_ds = UCF101Dataset(data_root, split='train', frames_per_clip=frames_per_clip, frame_step=frame_step, transform=transform)
    test_ds = UCF101Dataset(data_root, split='test', frames_per_clip=frames_per_clip, frame_step=frame_step, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    print(f"DataLoaders created: {len(train_ds)} train samples, {len(test_ds)} test samples, {num_classes} classes.")

    return train_loader, test_loader, num_classes, class_names