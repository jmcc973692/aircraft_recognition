import datetime
import os
import time
import traceback

import mlflow
import pandas as pd
import torch
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Set the flag to allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.AircraftBoundingDataset import AircraftBoundingDataset
from src.ImageTranformerAugment import ImageTransformerAugment
from src.ImageTransformer import ImageTransformer
from src.SSD512Model import SSD512Model
from src.TrainerSSD import TrainerSSD  # Import the Trainer class
from util.custom_collate import custom_collate_fn


def setup_device():
    """Setup CUDA device if available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename="checkpoint.pt"):
    """Save Training Checkpoints to Disk"""
    torch.save(state, filename)


def main():
    start_time = time.time()  # Record script start time

    # MLFlow Setup
    mlflow.pytorch.autolog()  # Automatically log important metrics
    mlflow.start_run()  # Start MLFlow run
    print("Started MLFlow run.")

    device = setup_device()  # Set up device
    print(f"Device setup complete: {device}")

    # Path to training images and bounding box labels
    img_dir = "data/image_dataset"
    csv_file = "data/all_dataset_bounding_boxes.csv"

    # Initialize the dataset with specific resizing and normalization
    transform_augment = ImageTransformerAugment(
        resize_dims=(512, 512), mean=[0.48236, 0.45882, 0.40784], std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
    )
    basic_transform = ImageTransformer(
        resize_dims=(512, 512), mean=[0.48236, 0.45882, 0.40784], std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
    )
    print("Initialized the ImageTransformer")

    # Read the CSV file
    annotations = pd.read_csv(csv_file)

    # Determine the most frequent class in each image for stratification
    annotations["most_frequent_class"] = annotations.groupby("filename")["class"].transform(lambda x: x.mode()[0])
    unique_files = annotations[["filename", "most_frequent_class"]].drop_duplicates()

    # Perform the stratified split
    train_files, valid_files = train_test_split(
        unique_files, test_size=0.2, random_state=42, stratify=unique_files["most_frequent_class"]
    )

    # Filter the original annotations to create training and validation sets
    train_annotations = annotations[annotations["filename"].isin(train_files["filename"])]
    valid_annotations = annotations[annotations["filename"].isin(valid_files["filename"])]

    # Directory to save the split data
    temp_dir = "temp"
    train_csv_path = os.path.join(temp_dir, "train_annotations.csv")
    valid_csv_path = os.path.join(temp_dir, "valid_annotations.csv")

    # Ensure the directory exists
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "checkpoints"), exist_ok=True)

    # Save the data
    train_annotations.to_csv(train_csv_path, index=False)
    valid_annotations.to_csv(valid_csv_path, index=False)
    print("Created Train and Validation Sets")

    # Initialize the model
    num_classes = 2  # Adjust based on the number of classes in your dataset
    model = SSD512Model(num_classes=num_classes, detection_threshold=0.4, iou_threshold=0.5, device=device)
    print("Initialized the Model")

    # Create the datasets with the new CSV files
    train_dataset = AircraftBoundingDataset(csv_file=train_csv_path, img_dir=img_dir, transform=transform_augment)
    valid_dataset = AircraftBoundingDataset(csv_file=valid_csv_path, img_dir=img_dir, transform=basic_transform)

    # Create DataLoaders for train and validation sets
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=28,
        shuffle=True,
        num_workers=9,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=28,
        shuffle=False,
        num_workers=9,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    print("DataLoaders Created")

    # Set up the optimizer
    learning_rate = 0.002  # Adjust as needed
    weight_decay = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.6)

    # Create an instance of with the model, device, and optimizer
    trainer = TrainerSSD(model, device, optimizer)

    # Initialize variables for tracking best validation loss and best model
    best_valid_loss = float("inf")
    # Date and timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # File path for saving the model with the date and timestamp
    model_path = f"models/detection_model_SSD512_{timestamp}.pth"

    # Epoch parameters
    max_epochs = 150  # Maximum number of epochs to train
    early_stop_patience = 8  # Number of epochs without improvement before stopping
    start_epoch = 0
    no_improvement_count = 0  # Counter for epochs without improvement

    # Check if a checkpoint exists
    checkpoint_path = "temp/checkpoints/SSD512_checkpoint.pt"
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

    print("Starting to Train!")
    # Training loop
    for epoch in range(start_epoch, max_epochs):
        start_train = time.time()  # Record training start time

        train_loss = trainer.train_epoch(train_loader)
        valid_loss = trainer.validate_epoch(valid_loader)

        scheduler.step(valid_loss)  # Update learning rate scheduler based on validation loss

        current_lr = scheduler.get_last_lr()

        print(f"Epoch {epoch + 1}/{max_epochs}, Current Learning Rate: {current_lr[0]:.6f}")

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)  # Save the best model
            no_improvement_count = 0  # Reset early stopping counter
        else:
            no_improvement_count += 1  # No improvement

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_valid_loss": best_valid_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "no_improvement_count": no_improvement_count,
            },
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
