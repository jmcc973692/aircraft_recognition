import os
import time

import mlflow
import mlflow.pytorch
import torch

from src.Config import Config
from src.DataHandler import DataHandler
from src.TorchConvNet import TorchConvNet
from src.Trainer import Trainer
from util.submissions import create_submission


def setup_device():
    """Setup CUDA device if available or CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    start_time = time.time()
    # MLFlow Setup: Starting the Run
    mlflow.start_run()
    print("Started MLFlow run.")

    # Initialize configuration and device setup
    config = Config(config_path="config.json")
    device = setup_device()
    print(f"Device setup complete: {device}")

    # Log configuration parameters
    mlflow.log_params(config.settings)
    print("Configuration parameters logged.")

    # Data transformations and handlers setup
    start_dh = time.time()
    data_handler = DataHandler(
        img_dir=config.get("img_dir"),
        transform=DataHandler.setup_transforms(),
        batch_size=config.get("batch_size"),
        num_workers=config.get("num_workers"),
        pin_memory=config.get("pin_memory"),
    )
    print(f"DataHandler setup completed in {time.time() - start_dh:.2f} seconds.")

    # Model and optimizer setup
    model = config.get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))
    print("Model and optimizer setup complete.")

    # Log the model
    mlflow.pytorch.log_model(model, "model", pip_requirements="requirements.txt")
    print("Model logged to MLFlow.")

    trainer = Trainer(model, device, optimizer)
    # Training loop
    for epoch in range(config.get("num_epochs")):
        start_train = time.time()
        train_loss = trainer.train_epoch(data_handler.trainloader)
        train_time = time.time() - start_train

        start_valid = time.time()
        valid_loss = trainer.validate_epoch(data_handler.validloader)
        valid_time = time.time() - start_valid

        print(f"Epoch {epoch+1}/{config.get('num_epochs')}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        print(f"Training time: {train_time:.2f}s, Validation time: {valid_time:.2f}s")

        # Log metrics to MLflow
        mlflow.log_metrics({"train_loss": train_loss, "valid_loss": valid_loss}, step=epoch)

    print("Training Complete.")

    # Setup test data loader and make predictions
    start_test_setup = time.time()
    testloader = data_handler.setup_test_loader(
        config.get("img_dir") + "/test_images",
        config.get("batch_size"),
        config.get("num_workers"),
        config.get("pin_memory"),
    )
    predictions = trainer.make_predictions(testloader)
    print(f"Test setup and prediction completed in {time.time() - start_test_setup:.2f} seconds.")

    # Ensure the directory exists
    submission_dir = "./submissions"
    os.makedirs(submission_dir, exist_ok=True)

    # Generate and save the submission file
    submission_filename = os.path.join(submission_dir, "submission.csv")
    create_submission(predictions, config.get("img_dir") + "/sample_submission.csv", submission_filename)
    print(f"Submission file saved: {submission_filename}")

    # End MLflow run
    mlflow.end_run()
    print(f"Total runtime: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execution was interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup code here, if any
        if "mlflow" in locals() or "mlflow" in globals():
            mlflow.end_run()
            print("MLFlow run ended.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
        print("Resources have been cleaned up.")
