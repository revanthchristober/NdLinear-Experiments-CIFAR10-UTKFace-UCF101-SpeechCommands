# ndlinear_project/cifar10/experiment.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from sklearn.metrics import classification_report, confusion_matrix

# Use relative imports within the package
from .model import NdLinearCIFAR10
from .dataset import get_cifar10_dataloaders
from ..utils.helpers import count_parameters, get_device # Example relative import

# --- Training and Evaluation Functions ---

def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def evaluate(model: nn.Module, device: torch.device, data_loader: DataLoader):
    """Evaluates the model and returns detailed metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    start_time = time.time()

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum').item() # Sum batch loss
            total_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

    inference_time = time.time() - start_time
    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    print(f'Evaluation: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    print(f"Total inference time: {inference_time:.2f}s")

    return avg_loss, accuracy, all_targets, all_preds, inference_time


# --- Main Experiment Runner ---

def run_cifar10_experiment():
    """Parses arguments and runs the full CIFAR10 experiment."""
    parser = argparse.ArgumentParser(description="CIFAR-10 with NdLinear")
    # Add arguments specific to CIFAR-10
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="test batch size")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--data_dir", type=str, default="./data", help="directory for storing data")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--report_file", type=str, default="cifar10_report.txt", help="path to save classification report")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/cifar10", help="directory to save checkpoints")
    # NdLinear specific params if needed, e.g.:
    # parser.add_argument("--nd_hidden_dims", type=str, default="4,4,128", help="Comma-separated hidden dims for NdLinear")

    # Use parse_known_args if called via main.py which might have its own args
    args, _ = parser.parse_known_args()

    # NdLinear hidden size processing (example if using arg)
    # try:
    #     nd_hidden_size = tuple(map(int, args.nd_hidden_dims.split(',')))
    # except ValueError:
    #     raise ValueError("Invalid format for --nd_hidden_dims. Use comma-separated integers like '4,4,128'")

    # Setup
    torch.manual_seed(args.seed)
    device = get_device(args.no_cuda)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    train_loader, test_loader = get_cifar10_dataloaders(
        args.data_dir, args.batch_size, args.test_batch_size, args.num_workers
    )

    # Model and Optimizer
    model = NdLinearCIFAR10().to(device) # Pass nd_hidden_size if needed
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model: NdLinearCIFAR10")
    print(f"Total trainable parameters: {count_parameters(model)}")

    # Training Loop
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)
        val_loss, val_acc, _, _, _ = evaluate(model, device, test_loader) # Using test set as validation

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            print(f"*** New best validation accuracy: {best_acc:.2f}% (Epoch {epoch}) ***")
            # Optionally save best model separately
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Saved best model checkpoint: {best_ckpt_path}")


    print(f"\nTraining finished. Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")

    # Final Evaluation (optionally load best model)
    print("\n--- Final Evaluation on Test Set ---")
    # To load best model:
    # best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    # if os.path.exists(best_model_path):
    #     print(f"Loading best model from {best_model_path}")
    #     model.load_state_dict(torch.load(best_model_path, map_location=device))
    # else:
    #     print("Best model checkpoint not found, evaluating model from last epoch.")

    test_loss, test_acc, targets, preds, inference_time = evaluate(model, device, test_loader)

    # Reporting
    report = classification_report(targets, preds, digits=4) # Add target_names if you have them mapped
    conf_mat = confusion_matrix(targets, preds)

    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", conf_mat)

    # Save final report
    try:
        with open(args.report_file, 'w') as f:
            f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"Final Test Loss: {test_loss:.4f}\n")
            f.write(f"Total Inference Time: {inference_time:.2f}s\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(conf_mat))
        print(f"Saved classification report to {args.report_file}")
    except IOError as e:
        print(f"Error saving report file: {e}")