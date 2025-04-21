# ndlinear_project/speech_commands/experiment.py
import torch
import torch.nn as nn
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

from .model import NdLinearAudio
from .dataset import get_speechcommands_dataloaders
from ..utils.helpers import count_parameters, get_device

# --- Training and Evaluation Functions ---

def train_epoch_speech(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epoch: int):
    """Trains the Speech Commands model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for batch_idx, (mels, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]")):
         # Skip batches with error labels (-1) from collate_fn
        if -1 in labels:
            print(f"Warning: Skipping training batch {batch_idx} due to invalid label.")
            continue

        mels, labels = mels.to(device), labels.to(device)
        batch_size = mels.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        logits = model(mels)
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

def evaluate_speech(model: nn.Module, device: torch.device, data_loader: DataLoader, criterion: nn.Module, num_classes: int, class_names: list, eval_type: str = "Eval"):
    """Evaluates the Speech Commands model on a given dataloader (val or test)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for mels, labels in tqdm(data_loader, desc=f"[{eval_type}]"):
             # Skip batches with error labels (-1) from collate_fn
            if -1 in labels:
                print(f"Warning: Skipping {eval_type} batch due to invalid label.")
                continue

            mels, labels = mels.to(device), labels.to(device)
            logits = model(mels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * mels.size(0)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    inference_time = time.time() - start_time
    num_eval_samples = len(all_labels)
    avg_loss = total_loss / num_eval_samples if num_eval_samples > 0 else 0
    throughput = num_eval_samples / inference_time if inference_time > 0 else 0

    if num_eval_samples == 0:
        print(f"Warning: No samples evaluated for {eval_type}. Returning empty results.")
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
         # Only calculate top-5 if k <= num_classes
         k_top = min(5, num_classes)
         topk_acc = top_k_accuracy_score(all_labels, all_probs_np, k=k_top, labels=range(num_classes))
         top5_acc = topk_acc if k_top == 5 else 0.0 # Store as top5 only if k was 5

    print(f"\n--- {eval_type} Results ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    if top5_acc > 0: print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Inference Throughput: {throughput:.2f} samples/sec")
    print(f"Total Inference Time: {inference_time:.2f}s")

    # Generate reports and CM
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    print("\nClassification Report:\n", report)
    # print("Confusion Matrix:\n", cm) # Usually okay for SpeechCommands

    # Per-class metrics for logging
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
    )

    metrics = {
        f"{eval_type.lower()}_avg_loss": avg_loss,
        f"{eval_type.lower()}_top1_acc": top1_acc,
        f"{eval_type.lower()}_top5_acc": top5_acc,
        f"{eval_type.lower()}_inference_throughput": throughput,
        f"{eval_type.lower()}_inference_time_sec": inference_time
    }
    # Add per-class metrics
    for idx, cls_name in enumerate(class_names):
        metrics[f"{eval_type.lower()}_precision_{cls_name}"] = precision[idx]
        metrics[f"{eval_type.lower()}_recall_{cls_name}"] = recall[idx]
        metrics[f"{eval_type.lower()}_f1_{cls_name}"] = f1[idx]

    reports_dict = {
        "classification_report": report,
        "confusion_matrix": cm
    }

    return metrics, reports_dict, cm


# --- Main Experiment Runner ---

def run_speech_commands_experiment():
    """Parses arguments and runs the full Speech Commands experiment."""
    parser = argparse.ArgumentParser(description="Speech Commands Classification with NdLinear")
    parser.add_argument("--data_path", type=str, default="./data/speech_commands", help="Path to download/load Speech Commands dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_mels", type=int, default=64, help="number of Mel frequency bins")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--report_dir", type=str, default="./reports/speech_commands", help="directory to save reports and plots")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/speech_commands", help="directory to save checkpoints")
    parser.add_argument("--mlflow_experiment_name", type=str, default="SpeechCommands-NdLinear", help="MLflow experiment name")
    # Add NdLinear specific args if needed

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = get_device(args.no_cuda)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # Data
    print("Loading Speech Commands data...")
    try:
        train_loader, val_loader, test_loader, num_classes, class_names = get_speechcommands_dataloaders(
            args.data_path, args.batch_size, args.num_workers, args.n_mels
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading data: {e}")
        return

    # Model and Optimizer
    model = NdLinearAudio(num_classes=num_classes, input_freq_bins=args.n_mels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss() # Add label smoothing?

    print(f"Model: NdLinearAudio ({num_classes} classes)")
    print(f"Total trainable parameters: {count_parameters(model)}")

    # MLflow Run
    with mlflow.start_run():
        print("Starting MLflow run...")
        mlflow.log_params(vars(args))
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("num_trainable_params", count_parameters(model))
        # Log class names as artifact?
        try:
            classes_path = os.path.join(args.report_dir, "classes.txt")
            with open(classes_path, "w") as f:
                 f.write("\n".join(class_names))
            mlflow.log_artifact(classes_path)
        except Exception as e:
            print(f"Could not save/log class names: {e}")


        # Training Loop
        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch_speech(
                model, device, train_loader, optimizer, criterion, epoch
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            # Validation loop
            val_metrics, _, _ = evaluate_speech(model, device, val_loader, criterion, num_classes, class_names, eval_type="Validation")
            if val_metrics: # Check if validation produced results
                mlflow.log_metrics({k: v for k, v in val_metrics.items() if 'loss' in k or 'acc' in k}, step=epoch) # Log key val metrics
                current_val_acc = val_metrics.get("validation_top1_acc", 0.0)

                # Save checkpoint
                ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)

                # Save best model based on validation accuracy
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"*** New best validation accuracy: {best_val_acc:.2f}%. Saved best model. ***")


        print("\nTraining finished.")

        # Final Evaluation on Test Set (Load best model)
        print("\n--- Final Evaluation on Test Set ---")
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            try:
                 model.load_state_dict(torch.load(best_model_path, map_location=device))
            except Exception as e:
                 print(f"Error loading best model state_dict: {e}. Evaluating model from last epoch.")
        else:
            print("Best model checkpoint not found, evaluating model from last epoch.")

        test_metrics, test_reports, test_cm = evaluate_speech(model, device, test_loader, criterion, num_classes, class_names, eval_type="Test")

        # Log final metrics
        if test_metrics:
             mlflow.log_metrics(test_metrics) # Log all test metrics

             # Save and log report
             report_path = os.path.join(args.report_dir, "test_classification_report.txt")
             with open(report_path, "w") as f:
                 f.write(test_reports["classification_report"])
             mlflow.log_artifact(report_path)
             print(f"Saved test classification report to {report_path}")

             # Save and log confusion matrix plot
             try:
                 plt.figure(figsize=(12, 10)) # Adjust size
                 sns.heatmap(test_cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                             annot=False, fmt="d", cbar=True) # Maybe skip annotation
                 plt.xticks(rotation=90, fontsize=8)
                 plt.yticks(rotation=0, fontsize=8)
                 plt.xlabel("Predicted")
                 plt.ylabel("True")
                 plt.title("Speech Commands Confusion Matrix (Test)")
                 plt.tight_layout()
                 cm_path = os.path.join(args.report_dir, "test_confusion_matrix.png")
                 plt.savefig(cm_path, dpi=150)
                 plt.close()
                 mlflow.log_artifact(cm_path)
                 print(f"Saved test confusion matrix plot to {cm_path}")
             except Exception as plot_err:
                 print(f"Error generating/saving confusion matrix plot: {plot_err}")

             # Log model (final or best)
             mlflow.pytorch.log_model(model, "final_model") # Log the evaluated model
             print("Logged final evaluated model to MLflow")

        print("MLflow run finished.")

# if __name__ == "__main__":
#     run_speech_commands_experiment()