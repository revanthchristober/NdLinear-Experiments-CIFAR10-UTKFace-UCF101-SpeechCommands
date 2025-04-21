# ndlinear_project/utkface/experiment.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score, f1_score
from tqdm import tqdm

from .model import NdLinearUTKFace
from .dataset import get_utkface_dataloaders
from ..utils.helpers import count_parameters, get_device

# --- Training and Evaluation Functions ---

def train_epoch_utkface(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int):
    """Trains the UTKFace model for one epoch."""
    model.train()
    total_loss_age, total_loss_gender, total_loss_race = 0.0, 0.0, 0.0
    num_samples = 0

    for batch_idx, (data, age_target, gender_target, race_target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]")):
        data = data.to(device)
        age_target = age_target.to(device)
        gender_target = gender_target.to(device) # Should be LongTensor
        race_target = race_target.to(device)     # Should be LongTensor
        num_samples += data.size(0)

        optimizer.zero_grad()
        age_pred, gender_pred_logits, race_pred_logits = model(data)

        # Calculate losses
        loss_age = F.mse_loss(age_pred.squeeze(1), age_target) # MSE for regression
        loss_gender = F.cross_entropy(gender_pred_logits, gender_target) # CE for classification
        loss_race = F.cross_entropy(race_pred_logits, race_target) # CE for classification

        # Combine losses (simple sum, could be weighted)
        total_loss = loss_age + loss_gender + loss_race

        total_loss.backward()
        optimizer.step()

        total_loss_age += loss_age.item() * data.size(0)
        total_loss_gender += loss_gender.item() * data.size(0)
        total_loss_race += loss_race.item() * data.size(0)

    avg_loss_age = total_loss_age / num_samples
    avg_loss_gender = total_loss_gender / num_samples
    avg_loss_race = total_loss_race / num_samples
    avg_total_loss = (total_loss_age + total_loss_gender + total_loss_race) / num_samples

    print(f"Epoch {epoch}: Train Avg Loss: {avg_total_loss:.4f} (Age: {avg_loss_age:.4f}, Gender: {avg_loss_gender:.4f}, Race: {avg_loss_race:.4f})")
    return avg_loss_age, avg_loss_gender, avg_loss_race

def evaluate_utkface(model: nn.Module, device: torch.device, test_loader: DataLoader, gender_names: list, race_names: list):
    """Evaluates the UTKFace model and returns detailed metrics."""
    model.eval()
    all_age_targets, all_age_preds = [], []
    all_gender_targets, all_gender_preds = [], []
    all_race_targets, all_race_preds = [], []
    start_time = time.time()

    with torch.no_grad():
        for data, age_target, gender_target, race_target in tqdm(test_loader, desc="[Eval]"):
            data = data.to(device)

            age_pred, gender_pred_logits, race_pred_logits = model(data)

            # Collect targets and predictions
            all_age_targets.extend(age_target.numpy())
            all_age_preds.extend(age_pred.squeeze(1).cpu().numpy())

            all_gender_targets.extend(gender_target.numpy())
            all_gender_preds.extend(gender_pred_logits.argmax(dim=1).cpu().numpy())

            all_race_targets.extend(race_target.numpy())
            all_race_preds.extend(race_pred_logits.argmax(dim=1).cpu().numpy())

    inference_time = time.time() - start_time

    # Calculate metrics
    test_age_mae = mean_absolute_error(all_age_targets, all_age_preds)
    test_gender_acc = accuracy_score(all_gender_targets, all_gender_preds)
    test_gender_f1 = f1_score(all_gender_targets, all_gender_preds, average='macro') # Use macro F1 for gender too
    test_race_acc = accuracy_score(all_race_targets, all_race_preds)
    test_race_f1 = f1_score(all_race_targets, all_race_preds, average='macro') # Macro F1

    print(f"\n--- Evaluation Results ---")
    print(f"Age MAE: {test_age_mae:.4f}")
    print(f"Gender Accuracy: {test_gender_acc:.4f}")
    print(f"Gender Macro F1: {test_gender_f1:.4f}")
    print(f"Race Accuracy: {test_race_acc:.4f}")
    print(f"Race Macro F1: {test_race_f1:.4f}")
    print(f"Total Inference Time: {inference_time:.2f}s")

    # Generate reports
    gender_report = classification_report(all_gender_targets, all_gender_preds, target_names=gender_names, digits=4)
    gender_cm = confusion_matrix(all_gender_targets, all_gender_preds)
    race_report = classification_report(all_race_targets, all_race_preds, target_names=race_names, digits=4)
    race_cm = confusion_matrix(all_race_targets, all_race_preds)

    print("\nGender Classification Report:\n", gender_report)
    print("Gender Confusion Matrix:\n", gender_cm)
    print("\nRace Classification Report:\n", race_report)
    print("Race Confusion Matrix:\n", race_cm)

    metrics = {
        "age_mae": test_age_mae,
        "gender_accuracy": test_gender_acc,
        "gender_f1_macro": test_gender_f1,
        "race_accuracy": test_race_acc,
        "race_f1_macro": test_race_f1,
        "inference_time_sec": inference_time
    }
    reports = {
        "gender_report": gender_report,
        "gender_cm": gender_cm,
        "race_report": race_report,
        "race_cm": race_cm
    }

    return metrics, reports


# --- Main Experiment Runner ---

def run_utkface_experiment():
    """Parses arguments and runs the full UTKFace experiment."""
    parser = argparse.ArgumentParser(description="UTKFace Age/Gender/Race Prediction with NdLinear")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root directory containing UTKFace jpg files (e.g., ./UTKFace)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training and testing")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--test_split", type=float, default=0.2, help="fraction of data for the test set")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--report_dir", type=str, default="./reports/utkface", help="directory to save classification reports")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/utkface", help="directory to save checkpoints")
    parser.add_argument("--mlflow_experiment_name", type=str, default="UTKFace-NdLinear", help="MLflow experiment name")
    # Add NdLinear specific args if needed

    args = parser.parse_args() # Use parse_args() when called directly via main.py

    # Setup
    torch.manual_seed(args.seed)
    device = get_device(args.no_cuda)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # Data
    print("Loading UTKFace data...")
    try:
        train_loader, test_loader, gender_names, race_names = get_utkface_dataloaders(
            args.data_root, args.batch_size, args.test_split, args.seed, args.num_workers
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the --data_root argument points to the correct directory.")
        return # Exit if data not found

    # Model and Optimizer
    model = NdLinearUTKFace().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model: NdLinearUTKFace")
    print(f"Total trainable parameters: {count_parameters(model)}")

    # MLflow Run
    with mlflow.start_run():
        print("Starting MLflow run...")
        # Log parameters
        mlflow.log_params(vars(args))
        mlflow.log_param("num_trainable_params", count_parameters(model))

        # Training Loop
        for epoch in range(1, args.epochs + 1):
            avg_loss_age, avg_loss_gender, avg_loss_race = train_epoch_utkface(
                model, device, train_loader, optimizer, epoch
            )
            # Log training losses
            mlflow.log_metric("train_loss_age", avg_loss_age, step=epoch)
            mlflow.log_metric("train_loss_gender", avg_loss_gender, step=epoch)
            mlflow.log_metric("train_loss_race", avg_loss_race, step=epoch)

            # Optional: Add validation loop here if needed

            # Save checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        print("\nTraining finished.")

        # Final Evaluation
        print("\n--- Final Evaluation on Test Set ---")
        metrics, reports = evaluate_utkface(model, device, test_loader, gender_names, race_names)

        # Log final metrics
        mlflow.log_metrics(metrics)

        # Save and log reports/artifacts
        try:
            gender_report_path = os.path.join(args.report_dir, "gender_report.txt")
            with open(gender_report_path, "w") as f:
                f.write(reports["gender_report"])
                f.write("\n\nConfusion Matrix:\n")
                f.write(np.array2string(reports["gender_cm"]))
            mlflow.log_artifact(gender_report_path)
            print(f"Saved gender report to {gender_report_path}")

            race_report_path = os.path.join(args.report_dir, "race_report.txt")
            with open(race_report_path, "w") as f:
                f.write(reports["race_report"])
                f.write("\n\nConfusion Matrix:\n")
                f.write(np.array2string(reports["race_cm"]))
            mlflow.log_artifact(race_report_path)
            print(f"Saved race report to {race_report_path}")

            # Log model
            mlflow.pytorch.log_model(model, "model")
            print("Logged model to MLflow")

        except Exception as e:
            print(f"Error saving reports or logging artifacts: {e}")

        print("MLflow run finished.")

# This allows running the script directly for debugging UTKFace
# if __name__ == "__main__":
#     run_utkface_experiment()