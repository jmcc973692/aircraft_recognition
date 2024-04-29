import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNAircraft:

    def __init__(self, num_classes, weights_path=None, device=None):
        """
        Initialize the Faster R-CNN model.
        :param num_classes: int, total number of classes including the background
        :param pretrained_path: str, path to the pretrained model weights, if available
        :param device: torch.device, device to run the model (CPU or GPU)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model = self.load_model(weights_path)

    def load_model(self, weights_path):
        """
        Load and modify the pre-trained Faster R-CNN model.
        :param pretrained_path: str, specifies the path to the pretrained model weights
        :return: model, a modified Faster R-CNN model
        """
        # Load the model without pretrained weights
        model = fasterrcnn_resnet50_fpn(weights=None)

        if weights_path:
            # Load the pretrained weights
            model.load_state_dict(torch.load(weights_path))

        # Modify the classifier to fit the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model.to(self.device)

    def predict(self, images):
        """
        Perform prediction on a list of images.

        :param images: list of PIL.Image, input images
        :return: list of dicts, each containing 'boxes', 'labels', and 'scores'
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Convert images to tensors and normalize
            transformed_images = [F.to_tensor(img).to(self.device) for img in images]
            transformed_images = [
                F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in transformed_images
            ]

            # Perform inference
            predictions = self.model(transformed_images)

            # Process predictions to make them more human-readable if necessary
            processed_predictions = [
                {
                    "boxes": pred["boxes"].cpu().numpy(),
                    "labels": pred["labels"].cpu().numpy(),
                    "scores": pred["scores"].cpu().numpy(),
                }
                for pred in predictions
            ]

            return processed_predictions
