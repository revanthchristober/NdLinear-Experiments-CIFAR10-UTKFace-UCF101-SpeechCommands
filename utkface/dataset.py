# ndlinear_project/utkface/dataset.py
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

class UTKFaceDataset(Dataset):
    """Custom Dataset for UTKFace images."""
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        file_name = os.path.basename(path)

        try:
            # Filename format: [age]_[gender]_[race]_[date&time].jpg
            age, gender, race = map(int, file_name.split("_")[:3])
            # Gender: 0 Male, 1 Female
            # Race: 0 White, 1 Black, 2 Asian, 3 Indian, 4 Others
        except ValueError:
            # Handle potential naming errors gracefully (e.g., assign defaults or skip)
            print(f"Warning: Skipping malformed filename: {file_name}")
            # Return a default/dummy sample or raise an error if preferred
            # For simplicity, returning zeros here, but might need better handling
            img = np.zeros((128, 128, 3), dtype=np.uint8) # Dummy image
            age, gender, race = 0, 0, 0
        else:
            try:
                img = cv2.imread(path)
                if img is None:
                    raise IOError(f"Could not read image: {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
                img = cv2.resize(img, (128, 128)) # Resize
            except Exception as e:
                 print(f"Error processing image {path}: {e}")
                 img = np.zeros((128, 128, 3), dtype=np.uint8) # Dummy image
                 age, gender, race = 0, 0, 0 # Reset labels for bad image

        if self.transform:
            # Apply transforms (expecting PIL or Tensor, handle numpy array)
            # ToTensor expects HWC uint8 or CHW float
            img_tensor = transforms.functional.to_tensor(img) # Converts HWC uint8 -> CHW float [0,1]
            img_tensor = self.transform(img_tensor) # Apply remaining transforms (like Normalize)
        else:
             img_tensor = transforms.functional.to_tensor(img)


        # Age as float for regression, gender/race as long for CrossEntropyLoss
        return img_tensor, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long), torch.tensor(race, dtype=torch.long)


def get_utkface_dataloaders(data_root: str, batch_size: int, test_split: float = 0.2, random_state: int = 42, num_workers: int = 4):
    """Finds images, splits data, and creates DataLoaders for UTKFace."""

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"UTKFace data directory not found: {data_root}")

    # Find all .jpg images recursively (adjust pattern if needed)
    # Example assumes images are directly in data_root like /path/to/UTKFace/*.jpg
    all_images = glob.glob(os.path.join(data_root, "*.jpg"))
    if not all_images:
         # Try searching in subdirs if the structure is different (e.g., utkface_aligned_cropped/UTKFace/*.jpg)
         all_images = glob.glob(os.path.join(data_root, "*", "*.jpg"))


    if not all_images:
        raise FileNotFoundError(f"No *.jpg images found in {data_root} or its immediate subdirectories.")

    print(f"Found {len(all_images)} images in {data_root}")

    # Define transforms
    transform = transforms.Compose([
        # ToTensor is handled inside Dataset's __getitem__ before other transforms
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])

    # Split data
    train_files, test_files = train_test_split(
        all_images, test_size=test_split, random_state=random_state
    )
    print(f"Train samples: {len(train_files)}, Test samples: {len(test_files)}")

    # Create datasets
    train_dataset = UTKFaceDataset(train_files, transform=transform)
    test_dataset = UTKFaceDataset(test_files, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Define target names for reporting
    gender_names = ["Male", "Female"]
    race_names = ["White", "Black", "Asian", "Indian", "Others"] # Match indices 0-4

    return train_loader, test_loader, gender_names, race_names