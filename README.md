
# NdLinear Project: Experiments with Vision and Audio

## Project Overview

This project was developed as part of the application for the **Ensemble Machine Learning Engineering Internship (Summer/Fall 2025)**. The core requirement was to build a project demonstrating the usage and capabilities of Ensemble's `NdLinear` layer, an N-dimensional linear layer designed to replace standard linear layers for potential benefits in model compression and efficiency.

The project explores the application of `NdLinear` across several standard machine learning tasks and datasets:

1.  **CIFAR-10:** Image classification using a simple CNN backbone where `NdLinear` replaces a traditional Flatten + Fully Connected layer.
2.  **UTKFace:** Multi-task facial attribute prediction (Age regression, Gender classification, Race classification) using a pre-trained ResNet backbone, with `NdLinear` processing the spatial features before task-specific prediction heads.
3.  **UCF101:** Video action recognition using a pre-trained ResNet backbone applied per-frame, where `NdLinear` operates on the resulting spatio-temporal feature maps.
4.  **Speech Commands:** Audio keyword classification using Mel Spectrograms as input, processed by a simple CNN backbone, followed by `NdLinear` operating on the pooled frequency-time features.

Experiments for UTKFace, UCF101, and Speech Commands leverage **MLflow** for tracking parameters, metrics, and artifacts.

Initially developed in a Google Colab notebook (`NdLinear-Experiments.ipynb` - if included), the code was subsequently refactored into a structured Python module (`ndlinear_project`) for better organization, reusability, and maintainability.

## Features

*   Implementation of `NdLinear` integration in various neural network architectures.
*   Experiments across image, video, and audio domains.
*   Multi-task learning example (UTKFace).
*   Spatiotemporal feature processing with `NdLinear` (UCF101).
*   Standard training and evaluation pipelines using PyTorch.
*   Detailed evaluation metrics including accuracy, loss, F1-score, MAE, classification reports, and confusion matrices.
*   MLflow integration for experiment tracking (UTKFace, UCF101, Speech Commands).
*   Modular code structure with clear separation of concerns (data, model, experiment logic).
*   Command-line interface for running individual experiments with configurable parameters.

## Project Structure

The project is organized into a Python package (`ndlinear_project`) and a main runner script.

```
ndlinear_project/
├── ndlinear_project/          # Main package directory
│   ├── __init__.py            # Makes the directory a Python package
│   ├── cifar10/               # CIFAR-10 experiment module
│   │   ├── __init__.py
│   │   ├── dataset.py         # Data loading functions and transforms
│   │   ├── model.py           # NdLinearCIFAR10 model definition
│   │   └── experiment.py      # Training/evaluation logic, main runner func
│   ├── utkface/               # UTKFace experiment module
│   │   ├── __init__.py
│   │   ├── dataset.py         # UTKFaceDataset class, data loading
│   │   ├── model.py           # NdLinearUTKFace model definition
│   │   └── experiment.py      # Training/evaluation/MLflow logic, runner func
│   ├── ucf101/                # UCF101 experiment module
│   │   ├── __init__.py
│   │   ├── dataset.py         # UCF101Dataset class, data loading
│   │   ├── model.py           # NdLinearUCF101 model definition
│   │   └── experiment.py      # Training/evaluation/MLflow logic, runner func
│   ├── speech_commands/       # Speech Commands experiment module
│   │   ├── __init__.py
│   │   ├── dataset.py         # SubsetSC class, collate_fn, data loading
│   │   ├── model.py           # NdLinearAudio model definition
│   │   └── experiment.py      # Training/evaluation/MLflow logic, runner func
│   └── utils/                 # Shared utility functions
│       ├── __init__.py
│       └── helpers.py         # e.g., count_parameters(), get_device()
├── main.py                    # Main entry point to select and run experiments
├── requirements.txt           # Python dependencies
├── checkpoints/               # Default directory for saving model checkpoints (created automatically)
├── data/                      # Default directory for downloading datasets (created automatically by torchvision/torchaudio)
├── reports/                   # Default directory for saving reports/plots (created automatically)
└── README.md                  # This file
```

*   **`ndlinear_project/ndlinear_project/`**: Contains the core source code organized by experiment.
*   **`model.py`**: Defines the PyTorch `nn.Module` incorporating `NdLinear` for that experiment.
*   **`dataset.py`**: Contains `Dataset` classes, transforms, and `DataLoader` creation logic.
*   **`experiment.py`**: Holds the training loop, evaluation functions, MLflow integration (if used), argument parsing, and the main `run_..._experiment()` function for that specific task.
*   **`utils/helpers.py`**: Common helper functions used across experiments.
*   **`main.py`**: The command-line entry point. Parses the desired experiment name and delegates execution to the corresponding `run_..._experiment()` function, passing along any remaining arguments.
*   **`requirements.txt`**: Lists all necessary Python packages.
*   **`checkpoints/`, `data/`, `reports/`**: Default output directories. These can often be overridden via command-line arguments.

## Setup Instructions

### 1. Prerequisites

*   Python (>= 3.8 recommended)
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### 2. Clone the Repository

```bash
git clone <your-repository-url> # Replace with the actual URL if hosted
cd ndlinear_project
```

### 3. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage dependencies cleanly.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 4. Install Dependencies

Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Note: If you encounter issues installing `av` or `opencv-python-headless`, you might need system-level libraries like FFmpeg or specific build tools, though `pip` usually handles this with pre-compiled wheels.)*

### 5. Data Acquisition

Some datasets are downloaded automatically, while others require manual steps:

*   **CIFAR-10:** Downloaded automatically by `torchvision` when the `cifar10` experiment is run for the first time. Data will typically be stored in `./data/cifar-10-batches-py/`.
*   **Speech Commands:** Downloaded automatically by `torchaudio` when the `speech_commands` experiment is run. Data will typically be stored in `./data/speech_commands/SpeechCommands/speech_commands_v0.02/`.
*   **UTKFace:** **Requires manual download and extraction.**
    *   Download from source (e.g., Kaggle: [https://www.kaggle.com/datasets/jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)). You might need a Kaggle account and API key setup, or download manually via browser.
    *   Example download command (requires Kaggle API setup):
        ```bash
        # Make sure kaggle api is installed and configured (pip install kaggle)
        # kaggle datasets download -d jangedoo/utkface-new -p ./temp_data --unzip
        # Or using curl if you found a direct link (less common for Kaggle)
        # curl -L -o ./temp_data/utkface-new.zip <direct_download_link>
        # unzip ./temp_data/utkface-new.zip -d ./data/UTKFace_Data
        ```
    *   **Crucially:** Ensure the final image files (`*.jpg`) are accessible. The code expects them either directly in the directory specified by `--data_root` or within an immediate subdirectory (like `UTKFace` inside the root). You **must** provide the correct path using the `--data_root` argument when running the experiment. Example: `--data_root ./data/UTKFace_Data/UTKFace` if the jpgs are in that specific subfolder.
*   **UCF101:** **Requires manual download and extraction.**
    *   Download from source (e.g., Kaggle: [https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition)).
    *   Example download command (requires Kaggle API setup):
        ```bash
        # kaggle datasets download -d matthewjansen/ucf101-action-recognition -p ./temp_data --unzip
        # Example manual unzip:
        # unzip ./temp_data/ucf101-action-recognition.zip -d ./data/UCF101_Data
        ```
    *   **Crucially:** The code expects a structure like `DATA_ROOT/train/ClassName/*.avi` and `DATA_ROOT/test/ClassName/*.avi`. Ensure your extracted data matches this structure. You **must** provide the path to the directory *containing* the `train/` and `test/` folders using the `--data_root` argument. Example: `--data_root ./data/UCF101_Data`.

## Usage Instructions

Experiments are run via the `main.py` script from the root directory of the project (`ndlinear_project/`).

### General Command Structure

```bash
python main.py <experiment_name> [experiment_specific_options...]
```

*   **`<experiment_name>`**: The name of the experiment to run. Choices are:
    *   `cifar10`
    *   `utkface`
    *   `ucf101`
    *   `speech_commands`
*   **`[experiment_specific_options...]`**: Command-line arguments specific to the chosen experiment (e.g., learning rate, epochs, data paths). These are defined within the respective `experiment.py` file using `argparse`. Use the `-h` or `--help` flag after the experiment name to see available options for that specific experiment (e.g., `python main.py cifar10 --help`).

### Examples

```bash
# === Run CIFAR-10 Experiment ===

# Run with default settings (checkpoints -> ./checkpoints/cifar10, report -> ./cifar10_report.txt)
python main.py cifar10

# Run with custom epochs, learning rate, and checkpoint directory
python main.py cifar10 --epochs 5 --lr 0.0005 --checkpoint_dir ./my_cifar_checkpoints


# === Run UTKFace Experiment ===

# Run with required data path (MLflow tracking enabled by default)
# Ensure /path/to/UTKFace contains the .jpg files (or subfolder like UTKFace/)
python main.py utkface --data_root /path/to/UTKFace

# Run with custom epochs and report directory
python main.py utkface --data_root /path/to/UTKFace --epochs 10 --report_dir ./my_utk_reports


# === Run UCF101 Experiment ===

# Run with required data path (MLflow tracking enabled by default)
# Ensure /path/to/UCF101 contains train/ and test/ subdirectories
python main.py ucf101 --data_root /path/to/UCF101

# Run with different number of frames per clip and batch size
python main.py ucf101 --data_root /path/to/UCF101 --frames_per_clip 8 --batch_size 4


# === Run Speech Commands Experiment ===

# Run with default settings (data downloaded to ./data/speech_commands if needed)
python main.py speech_commands

# Run with custom epochs and MLflow experiment name
python main.py speech_commands --epochs 15 --mlflow_experiment_name "My_Speech_Test" --data_path ./custom_audio_data

```

### Viewing MLflow Results

For experiments using MLflow (UTKFace, UCF101, Speech Commands), an `mlruns` directory will be created in the project root to store tracking data. To view the results in a web UI:

1.  Make sure MLflow is installed (`pip install mlflow`).
2.  Navigate to the project root directory (`ndlinear_project/`) in your terminal.
3.  Run the command:
    ```bash
    mlflow ui
    ```
4.  Open your web browser and go to `http://localhost:5000` (or the address shown in the terminal). You can then explore logged parameters, metrics, plots, reports, and saved models for each run.

### Outputs

*   **Checkpoints:** Model weights (`.pth` files) are saved during training in the specified checkpoint directory (default: `./checkpoints/<experiment_name>/`). The best model based on validation performance (if applicable) is often saved as `best_model.pth`.
*   **Reports:** Classification reports (`.txt`) and confusion matrix plots (`.png`) are saved in the specified report directory (default: `./reports/<experiment_name>/`).
*   **MLflow Artifacts:** For experiments using MLflow, reports, plots, and potentially the model itself are also logged as artifacts within the MLflow run, accessible via the MLflow UI.

---