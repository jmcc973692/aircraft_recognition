import os

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
    # MLFlow Setup: Starting the Run
    mlflow.start_run()

    # Initialize configuration and device setup
    config = Config(config_path="config.json")
    device = setup_device()

    # Log configuration parameters
    mlflow.log_params(config.settings)

    # Data transformations and handlers setup
    data_handler = DataHandler(
        img_dir=config.get("img_dir"),
        transform=DataHandler.setup_transforms(),
        batch_size=config.get("batch_size"),
        num_workers=config.get("num_workers"),
        pin_memory=config.get("pin_memory"),
    )

    # Model and optimizer setup
    model = TorchConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))

    # Log the model
    mlflow.pytorch.log_model(model, "model", pip_requirements="requirements.txt")

    trainer = Trainer(model, device, optimizer)

    # Training loop
    for epoch in range(config.get("num_epochs")):
        train_loss = trainer.train_epoch(data_handler.trainloader)
        valid_loss = trainer.validate_epoch(data_handler.validloader)
        print(f"Epoch {epoch+1}/{config.get('num_epochs')}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics({"train_loss": train_loss, "valid_loss": valid_loss}, step=epoch)

    print("Training Complete")

    # Setup test data loader and make predictions
    testloader = data_handler.setup_test_loader(
        config.get("img_dir") + "/test_images",
        config.get("batch_size"),
        config.get("num_workers"),
        config.get("pin_memory"),
    )
    predictions = trainer.make_predictions(testloader)

    # Ensure the directory exists
    submission_dir = "./submissions"
    os.makedirs(submission_dir, exist_ok=True)

    # Generate and save the submission file with a specified filename
    submission_filename = os.path.join(submission_dir, "submission.csv")
    create_submission(predictions, config.get("img_dir") + "/sample_submission.csv", submission_filename)

    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
