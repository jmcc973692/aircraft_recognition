import datetime
import os
import time

import mlflow
import pandas as pd
import torch
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Set the flag to allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.AircraftBoundingDataset import AircraftBoundingDataset
from src.FasterRCNNAircraft import FasterRCNNAircraft
from src.ImageTransformer import ImageTransformer
from src.SSDModel import SSDModel
from src.TrainerSSDLite import TrainerSSDLite  # Import the Trainer class
from util.custom_collate import custom_collate_fn


def setup_device():
    """Setup CUDA device if available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    transform = ImageTransformer()  # Apply resizing and normalization

    # Read the CSV file and group by 'filename'
    annotations = pd.read_csv(csv_file)
    grouped_annotations = annotations.groupby("filename")

    # Create separate DataFrames for filenames and annotations
    filename_df = pd.DataFrame({"filename": grouped_annotations.groups.keys()})

    # Split by filename to keep all bounding boxes for a given image together
    train_filenames, valid_filenames = train_test_split(filename_df, test_size=0.2, random_state=42)  # 80-20 split

    # Merge to get the corresponding annotations for each subset
    train_annotations = annotations[annotations["filename"].isin(train_filenames["filename"])]
    valid_annotations = annotations[annotations["filename"].isin(valid_filenames["filename"])]

    # Save split data to temporary CSV files
    train_csv_path = "temp/train_annotations.csv"
    valid_csv_path = "temp/valid_annotations.csv"

    # Ensure the temp directory exists
    os.makedirs("temp", exist_ok=True)
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Save the data
    train_annotations.to_csv(train_csv_path, index=False)
    valid_annotations.to_csv(valid_csv_path, index=False)

    # Initialize the model
    num_classes = 2  # Adjust based on the number of classes in your dataset
    model = SSDModel(num_classes=num_classes, detection_threshold=0.3, device=device)
    print("Initialized the Model")

    # Create the datasets with the new CSV files
    train_dataset = AircraftBoundingDataset(csv_file=train_csv_path, img_dir=img_dir, transform=transform)
    valid_dataset = AircraftBoundingDataset(csv_file=valid_csv_path, img_dir=img_dir, transform=transform)

    # Create DataLoaders for train and validation sets
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=7, pin_memory=True, collate_fn=custom_collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, shuffle=False, num_workers=7, pin_memory=True, collate_fn=custom_collate_fn
    )
    print("DataLoaders Created")

    # Set up the optimizer
    learning_rate = 1e-4  # Adjust as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of TrainerFasterRCNN with the model, device, and optimizer
    trainer = TrainerSSDLite(model, device, optimizer)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.1)

    # Create an instance of TrainerSSDLite with the model, device, and optimizer
    trainer = TrainerSSDLite(model, device, optimizer)

    # Initialize variables for tracking best validation loss and best model
    best_valid_loss = float("inf")
    # Date and timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # File path for saving the model with the date and timestamp
    model_path = f"models/detection_model_{timestamp}.pth"

    # Epoch parameters
    max_epochs = 150  # Maximum number of epochs to train
    early_stop_patience = 5  # Number of epochs without improvement before stopping

    no_improvement_count = 0  # Counter for epochs without improvement
    print("Starting to Train!")
    # Training loop
    for epoch in range(max_epochs):
        start_train = time.time()  # Record training start time

        train_loss = trainer.train_epoch(train_loader)
        valid_loss = trainer.validate_epoch(valid_loader)

        scheduler.step(valid_loss)  # Update learning rate scheduler based on validation loss

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)  # Save the best model
            no_improvement_count = 0  # Reset early stopping counter
        else:
            no_improvement_count += 1  # No improvement

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
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache if available
        if "mlflow" in locals():
            mlflow.end_run()  # End MLFlow run
            print("Resources have been cleaned up.")  # General cleanup statement
