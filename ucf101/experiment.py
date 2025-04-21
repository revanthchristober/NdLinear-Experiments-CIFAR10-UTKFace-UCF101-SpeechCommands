# ndlinear_project/ucf101/experiment.py
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             top_k_accuracy_score, precision_recall_fscore_support)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .model import NdLinearUCF101
from .dataset import get_ucf101_dataloaders
from ..utils.helpers import count_parameters, get_device

# --- Training and Evaluation Functions ---

def train_epoch_ucf101(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epoch: int):
    """Trains the UCF101 model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for batch_idx, (clips, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]")):
        # Skip potentially problematic batches from dataset loading errors
        if -1 in labels:
            print(f"Warning: Skipping batch {batch_idx} due to data loading error.")
            continue

        clips, labels = clips.to(device), labels.to(device)
        batch_size = clips.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct_predictions += preds.eq(labels).sum().item()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0
    print(f"Epoch {epoch}: Train Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def evaluate_ucf101(model: nn.Module, device: torch.device, test_loader: DataLoader, criterion: nn.Module, num_classes: int, class_names: list):
    """Evaluates the UCF101 model and returns detailed metrics."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc="[Eval]"):
             # Skip potentially problematic batches from dataset loading errors
            if -1 in labels:
                print(f"Warning: Skipping evaluation batch due to data loading error.")
                continue

            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            total_loss += loss.item() * clips.size(0)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs) # Collect probabilities for top-k

    inference_time = time.time() - start_time
    num_test_samples = len(all_labels)
    avg_loss = total_loss / num_test_samples if num_test_samples > 0 else 0
    throughput = num_test_samples / inference_time if inference_time > 0 else 0

    if num_test_samples == 0:
        print("Warning: No samples evaluated. Returning empty results.")
        return {}, {}, None # Return empty dicts/None

    # Calculate metrics
    top1_acc = accuracy_score(all_labels, all_preds)
    # Ensure all_probs is a numpy array for top_k
    all_probs_np = np.array(all_probs)
    # Check if shape is correct (N_samples, N_classes)
    if all_probs_np.ndim != 2 or all_probs_np.shape[1] != num_classes:
         print(f"Warning: Probability array shape mismatch ({all_probs_np.shape}). Cannot calculate top-5 accuracy.")
         top5_acc = 0.0
    else:
        top5_acc = top_k_accuracy_score(all_labels, all_probs_np, k=5, labels=range(num_classes))

    print(f"\n--- Evaluation Results ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Inference Throughput: {throughput:.2f} clips/sec")
    print(f"Total Inference Time: {inference_time:.2f}s")

    # Generate reports and CM
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    print("\nClassification Report:\n", report)
    # print("Confusion Matrix:\n", cm) # Can be very large

    # Per-class metrics for logging
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
    )

    metrics = {
        "test_avg_loss": avg_loss,
        "test_top1_acc": top1_acc,
        "test_top5_acc": top5_acc,
        "inference_throughput_clips_per_sec": throughput,
        "inference_time_sec": inference_time
    }
    # Add per-class metrics dynamically
    for idx, cls_name in enumerate(class_names):
        metrics[f"precision_{cls_name}"] = precision[idx]
        metrics[f"recall_{cls_name}"] = recall[idx]
        metrics[f"f1_{cls_name}"] = f1[idx]

    reports_dict = {
        "classification_report": report,
        "confusion_matrix": cm
    }

    return metrics, reports_dict, cm # Return CM separately for plotting


# --- Main Experiment Runner ---

def run_ucf101_experiment():
    """Parses arguments and runs the full UCF101 experiment."""
    parser = argparse.ArgumentParser(description="UCF101 Action Recognition with NdLinear")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root directory containing UCF101 train/test splits (e.g., ./UCF101)")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--frames_per_clip", type=int, default=4, help="number of frames per video clip")
    parser.add_argument("--frame_step", type=int, default=2, help="step between sampled frames")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--report_dir", type=str, default="./reports/ucf101", help="directory to save reports and plots")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ucf101", help="directory to save checkpoints")
    parser.add_argument("--mlflow_experiment_name", type=str, default="UCF101-NdLinear", help="MLflow experiment name")
    # Add NdLinear specific args if needed

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = get_device(args.no_cuda)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # Data
    print("Loading UCF101 data...")
    try:
        train_loader, test_loader, num_classes, class_names = get_ucf101_dataloaders(
            args.data_root, args.frames_per_clip, args.batch_size, args.frame_step, args.num_workers
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the --data_root argument points to the directory containing train/ and test/ subfolders.")
        return

    # Model and Optimizer
    model = NdLinearUCF101(num_classes=num_classes, frames_per_clip=args.frames_per_clip).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Model: NdLinearUCF101 ({num_classes} classes)")
    print(f"Total trainable parameters: {count_parameters(model)}")

    # MLflow Run
    with mlflow.start_run():
        print("Starting MLflow run...")
        mlflow.log_params(vars(args))
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("num_trainable_params", count_parameters(model))

        # Training Loop
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch_ucf101(
                model, device, train_loader, optimizer, criterion, epoch
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            # Optional: Validation loop
            # if val_loader: evaluate_ucf101(...)

            # Save checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            # mlflow.log_artifact(ckpt_path, artifact_path="checkpoints") # Optionally log checkpoints

        print("\nTraining finished.")

        # Final Evaluation
        print("\n--- Final Evaluation on Test Set ---")
        metrics, reports, cm = evaluate_ucf101(model, device, test_loader, criterion, num_classes, class_names)

        # Log final metrics
        if metrics: # Check if evaluation produced results
             mlflow.log_metrics(metrics)

             # Save and log report
             report_path = os.path.join(args.report_dir, "classification_report.txt")
             with open(report_path, "w") as f:
                 f.write(reports["classification_report"])
             mlflow.log_artifact(report_path)
             print(f"Saved classification report to {report_path}")

             # Save and log confusion matrix plot (optional due to size)
             try:
                 plt.figure(figsize=(20, 18)) # Adjust size as needed
                 sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, fmt="d", cbar=False)
                 plt.xlabel("Predicted")
                 plt.ylabel("True")
                 plt.title("UCF101 Confusion Matrix")
                 plt.tight_layout()
                 cm_path = os.path.join(args.report_dir, "confusion_matrix.png")
                 plt.savefig(cm_path, dpi=150)
                 plt.close()
                 mlflow.log_artifact(cm_path)
                 print(f"Saved confusion matrix plot to {cm_path}")
             except Exception as plot_err:
                 print(f"Error generating/saving confusion matrix plot: {plot_err}")


             # Log model
             mlflow.pytorch.log_model(model, "model")
             print("Logged model to MLflow")

        print("MLflow run finished.")

# if __name__ == "__main__":
#     run_ucf101_experiment()