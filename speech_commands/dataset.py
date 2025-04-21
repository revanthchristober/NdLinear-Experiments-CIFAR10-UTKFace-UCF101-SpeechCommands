# ndlinear_project/speech_commands/dataset.py
import os
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader

class SubsetSC(SPEECHCOMMANDS):
    """Helper class to get official splits of SpeechCommands."""
    def __init__(self, data_path:str, subset: str = None):
        # data_path is where to download/look for the dataset
        super().__init__(data_path, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                # Return full paths relative to the dataset root
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            # Include only files NOT in validation or testing sets
            # Need to walk the full dataset path to get full paths for comparison
            full_walker = [os.path.join(self._path, self._relpath, p) for p in super()._load_list(self._ext_audio)]
            self._walker = [w for w in full_walker if w not in excludes]
        elif subset is None:
            # Use the full dataset walker from parent if subset is None
             full_walker = [os.path.join(self._path, self._relpath, p) for p in super()._load_list(self._ext_audio)]
             self._walker = full_walker
        else:
             raise ValueError(f"Unknown subset '{subset}'. Must be 'training', 'validation', 'testing', or None.")

        # Ensure walker contains full paths
        # print(f"Subset '{subset}' - first few paths: {self._walker[:5]}")


# Define the Collate Function globally or inside get_dataloaders
def make_collate_fn(labels_list, sample_rate=16000, n_mels=64, target_len_samples=16000):
    """Creates a collate function that uses the provided labels list."""
    label_to_index = {label: i for i, label in enumerate(labels_list)}

    def collate_fn(batch):
        tensors, targets = [], []
        new_sample_rate = sample_rate

        # Pre-create transforms for efficiency
        resampler_cache = {} # Cache resamplers based on original freq
        melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=new_sample_rate, n_mels=n_mels, n_fft=400, hop_length=160 # Example params
        ).to(batch[0][0].device if torch.cuda.is_available() else 'cpu') # Move transform to device if possible

        for waveform, orig_sample_rate, label, speaker_id, utterance_num in batch:
            # 1. Resample if necessary
            if orig_sample_rate != new_sample_rate:
                if orig_sample_rate not in resampler_cache:
                    resampler_cache[orig_sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=orig_sample_rate, new_freq=new_sample_rate
                    ).to(waveform.device)
                waveform = resampler_cache[orig_sample_rate](waveform)

            # 2. Pad/Truncate to target length (1 second = 16000 samples at 16kHz)
            current_len = waveform.size(1)
            if current_len < target_len_samples:
                padding = target_len_samples - current_len
                waveform = F.pad(waveform, (0, padding)) # Pad at the end
            elif current_len > target_len_samples:
                waveform = waveform[:, :target_len_samples] # Truncate

            # 3. Compute Mel Spectrogram
            # Ensure waveform is on the same device as the transform
            mel = melspec_transform(waveform.to(melspec_transform.device)) # [1, n_mels, time_steps]

            # Apply log scale (common practice)
            mel = torchaudio.transforms.AmplitudeToDB()(mel)

            tensors.append(mel.squeeze(0)) # Remove channel dim -> [n_mels, time_steps]

            # 4. Map label string to index
            if label not in label_to_index:
                 print(f"Warning: Label '{label}' not in known labels list. Assigning index -1.")
                 targets.append(-1) # Or handle unknown labels differently
            else:
                 targets.append(label_to_index[label])

        # Stack tensors: list of [F, T] -> [B, F, T] -> add channel dim [B, 1, F, T]
        tensors_stacked = torch.stack(tensors).unsqueeze(1)
        targets_tensor = torch.tensor(targets, dtype=torch.long)

        return tensors_stacked, targets_tensor

    return collate_fn


def get_speechcommands_dataloaders(data_path: str = "./data/speech_commands", batch_size: int = 64, num_workers: int = 4, n_mels: int = 64, sample_rate: int = 16000):
    """Creates DataLoaders for the Google Speech Commands dataset."""

    # Instantiate datasets for each split
    train_set = SubsetSC(data_path, "training")
    val_set = SubsetSC(data_path, "validation")
    test_set = SubsetSC(data_path, "testing")

    # Determine all unique labels from the training set
    # Handle potential variations in how labels are stored/accessed
    try:
        # Accessing the label string directly might depend on torchaudio version
        # This assumes the standard format where label is the 3rd element
        labels = sorted(list(set(sample[2] for sample in train_set)))
    except IndexError:
        # Fallback if format changed (less likely for standard datasets)
        print("Warning: Could not automatically determine labels. Using a predefined list (might be incomplete/incorrect).")
        # Provide a default list or raise error
        # labels = ['yes', 'no', ...] # Example placeholder
        raise RuntimeError("Failed to determine class labels automatically.")

    if not labels:
        raise RuntimeError("No labels found in the training set.")

    print(f"Found {len(labels)} classes: {labels[:5]}...") # Print first few labels

    num_classes = len(labels)

    # Create the collate function using the determined labels
    collate = make_collate_fn(labels, sample_rate=sample_rate, n_mels=n_mels)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, num_workers=num_workers, pin_memory=True)

    print(f"DataLoaders created: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test samples.")

    return train_loader, val_loader, test_loader, num_classes, labels