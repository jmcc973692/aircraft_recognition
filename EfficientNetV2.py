import datetime
import os
import time
import traceback

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.CroppedDataset import CroppedDataset
from src.CroppedImageTransformer import CroppedImageTransformer
from src.EfficientNetV2MClassifier import EfficientNetV2MClassifier
from src.TrainerEffNet import TrainerEffNet


def load_dataset(base_path="input/crop"):
    images = []
    labels = []
    label_map = {}  # Map class names to integer labels

    for i, class_name in enumerate(os.listdir(base_path)):
        class_path = os.path.join(base_path, class_name)
        label_map[class_name] = i
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            images.append(img_path)
            labels.append(i)

    return images, labels, label_map


def split_data(images, labels, test_size=0.1, val_size=0.15):
    # First split to separate out the test set
    images_train_val, images_test, labels_train_val, labels_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # Adjust val_size to account for the reduced number of train+val samples
    val_size_adjusted = val_size / (1 - test_size)

    # Split the remaining data into training and validation sets
    images_train, images_val, labels_train, labels_val = train_test_split(
        images_train_val, labels_train_val, test_size=val_size_adjusted, stratify=labels_train_val, random_state=42
    )

    return images_train, images_val, images_test, labels_train, labels_val, labels_test


def setup_device():
    """Setup CUDA device if available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, scheduler, filename="checkpoint.pth.tar"):
    """
    Save the training checkpoint along with the state of two schedulers.
    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, best valid loss, etc.
        scheduler_warmup (torch.optim.lr_scheduler): The warmup scheduler.
        scheduler_cosine (torch.optim.lr_scheduler): The cosine annealing scheduler.
        filename (str): Path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save scheduler states within the main state dictionary
    state["scheduler"] = scheduler.state_dict()

    torch.save(state, filename)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load the model, optimizer, and scheduler states from a checkpoint.
    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        scheduler_warmup (torch.optim.lr_scheduler): The warmup scheduler to load the state into.
        scheduler_cosine (torch.optim.lr_scheduler): The cosine annealing scheduler to load the state into.
        checkpoint_path (str): Path to the checkpoint file.
    """
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        best_valid_loss = checkpoint["best_valid_loss"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        no_improvement_count = checkpoint["no_improvement_count"]
        print(f"=> loaded checkpoint '{checkpoint_path}' - epoch {checkpoint['epoch']}")
        return start_epoch, best_valid_loss, no_improvement_count
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float("inf"), 0


# Warmup Scheduler
def warmup_scheduler(epoch):
    if epoch < 6:
        return float(epoch) / 6
    return 1


def evaluate_model(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Outputs are probabilities
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate metrics
    accuracy = accuracy_score(np.round(all_targets), np.round(all_preds))
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, np.round(all_preds), average="macro")
    auc_roc = roc_auc_score(all_targets, all_preds, multi_class="ovr")

    return accuracy, precision, recall, f1_score, auc_roc


def main():
    start_time = time.time()  # Record script start time

    # MLFlow Setup
    mlflow.pytorch.autolog()  # Automatically log important metrics
    mlflow.start_run()  # Start MLFlow run
    print("Started MLFlow run.")

    device = setup_device()  # Set up device
    print(f"Device setup complete: {device}")

    # Load dataset
    images, labels, label_map = load_dataset(base_path="input/crop")

    # Split dataset
    images_train, images_val, images_test, labels_train, labels_val, labels_test = split_data(images, labels)

    print("Training set size:", len(images_train))
    print("Validation set size:", len(images_val))
    print("Test set size:", len(images_test))
    print(f"Class Mapping: {label_map}")

    # Initialize Transformers
    basic_transform = CroppedImageTransformer(resize_dims=(480, 480))
    transform_augment = CroppedImageTransformer(resize_dims=(480, 480), augment=True)
    print("Initialized the ImageTransformers")

    # Create Datasets
    train_dataset = CroppedDataset(images=images_train, labels=labels_train, transformer=transform_augment)
    valid_dataset = CroppedDataset(images=images_val, labels=labels_val, transformer=basic_transform)
    print("Initialized the Cropped Datasets")

    # Create DataLoaders for the train and validation sets
    train_loader = DataLoader(
        train_dataset, batch_size=6, shuffle=True, num_workers=9, pin_memory=True, prefetch_factor=2
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        prefetch_factor=2,
    )
    print("DataLoaders Created")

    # Initialize the Model
    model = EfficientNetV2MClassifier(num_classes=16, device=device)
    print("Initialized the Model")

    # Setup Optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # Setup Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    trainer = TrainerEffNet(model, device, optimizer)

    best_valid_loss = float("inf")
    no_improvement_count = 0
    start_epoch = 0
    early_stop_patience = 10
    max_epochs = 150

    # File path for saving the model with the date and timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"models/classification_model_EffNetV2M_{timestamp}.pth"

    # Check if a checkpoint exists and Load it
    checkpoint_path = "temp/checkpoints/EffNetV2M_checkpoint.pt"
    start_epoch, best_valid_loss, no_improvement_count = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    print("Starting to Train!")
    for epoch in range(start_epoch, max_epochs):
        start_train = time.time()  # Record training start time

        train_loss = trainer.train_epoch(train_loader)
        valid_loss = trainer.validate_epoch(valid_loader)

        scheduler.step(valid_loss)
        current_lr = scheduler.get_last_lr()

        print(f"Epoch {epoch + 1}/{max_epochs}, Current Learning Rate: {current_lr[0]:.6f}")

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)  # Save the best model
            no_improvement_count = 0  # Reset early stopping counter
        else:
            no_improvement_count += 1  # Increment no improvement count

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_valid_loss": best_valid_loss,
                "optimizer": optimizer.state_dict(),
                "no_improvement_count": no_improvement_count,
            },
            scheduler,
            filename=checkpoint_path,
        )

        train_time = time.time() - start_train
        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        print(f"Training time: {train_time:.2f} seconds.")

        # Check if we should stop early
        if no_improvement_count >= early_stop_patience:
            print(f"No improvement for {early_stop_patience} epochs. Stopping training.")
            break  # Early stopping

    # Load the best model after training
    model.load_state_dict(torch.load(model_path))

    # Evaluate the Model on the Test Set
    # Create the Test Dataset and DataLoader
    test_transformer = CroppedImageTransformer(resize_dims=(480, 480))  # No augmentation for testing
    test_dataset = CroppedDataset(images=images_test, labels=labels_test, transformer=test_transformer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=9, pin_memory=True)

    # Evaluate the model on the test set
    accuracy, precision, recall, f1_score, auc_roc = evaluate_model(model, device, test_loader)
    print(
        f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, AUC-ROC: {auc_roc:.4f}"
    )

    # End MLFlow run
    mlflow.end_run()  # End the MLFlow run

    # Total runtime
    total_runtime = time.time() - start_time
    print(f"Script execution completed in {total_runtime:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()  # Execute the main function
    except KeyboardInterrupt:
        print("Execution was interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()  # Print the stack trace
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache if available
        if "mlflow" in locals():
            mlflow.end_run()  # End MLFlow run
            print("Resources have been cleaned up.")  # General cleanup statement
