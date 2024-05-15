import os
import re

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor

from src.CroppedImageTransformer import CroppedImageTransformer
from src.EfficientNetB4Classifier import EfficientNetB4Classifier
from src.EfficientNetV2MClassifier import EfficientNetV2MClassifier
from src.ImageTransformer import ImageTransformer
from src.SSD512Model import SSD512Model
from util.submissions import create_submission

CLASS_MAPPING = {
    "A10": 0,
    "B1": 1,
    "B2": 2,
    "B52": 3,
    "C130": 4,
    "C17": 5,
    "E2": 6,
    "F14": 7,
    "F15": 8,
    "F16": 9,
    "F18": 10,
    "F22": 11,
    "F35": 12,
    "F4": 13,
    "V22": 14,
    "xBackground": 15,
}


def combine_probabilities(probs):
    """Combine probabilities using logical OR approach."""
    return 1 - np.prod([1 - p for p in probs])


# Setup and Model Init #################################################################################################

# Path setup
detection_model_path = "models/detection_SSD512_best.pth"
classification_model_path = "models/classification_EffNetB4_best.pth"
test_images_dir = "input/test_images"
cropped_images_dir = "temp/cropped_images"

# Ensure output directory exists
os.makedirs(cropped_images_dir, exist_ok=True)

# Load the primary SSD model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detection_model_1 = SSD512Model(
    num_classes=2, detection_threshold=0.4, iou_threshold=0.5, device=device
)  # Default thresholds
detection_model_1.load_state_dict(torch.load(detection_model_path))
detection_model_1.eval()
detection_model_1.to(device)

# Load the fallback SSD model with a lower detection threshold
detection_model_2 = SSD512Model(
    num_classes=2, detection_threshold=0.14, iou_threshold=0.75, device=device
)  # Lower threshold
detection_model_2.load_state_dict(torch.load(detection_model_path))
detection_model_2.eval()
detection_model_2.to(device)

# Load the Classification Model
classification_model = EfficientNetB4Classifier(num_classes=16, device=device)
classification_model.load_state_dict(torch.load(classification_model_path))
classification_model.eval()
classification_model.to(device)

# Initialize Image transformer
detection_transformer = ImageTransformer(
    resize_dims=(512, 512), mean=[0.48236, 0.45882, 0.40784], std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
)
classification_transformer = CroppedImageTransformer(resize_dims=(380, 380), augment=False)

# Image Aircraft Detection #############################################################################################

test_images = os.listdir(test_images_dir)

# Counters
secondary_model_used = 0
third_model_used = 0
no_detections = 0
no_detection_images = []
detection_scores = {}

# Process each test image for detection and cropping
# Process each test image for detection and cropping
for img_name in test_images:
    img_path = os.path.join(test_images_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    original_width, original_height = image.size
    transformed_image = detection_transformer(image)
    input_tensor = transformed_image.unsqueeze(0).to(device)

    # Attempt with first model
    with torch.no_grad():
        primary_outputs = detection_model_1(input_tensor)
    if len(primary_outputs[0]["boxes"]) == 0:
        secondary_model_used += 1
        with torch.no_grad():
            fallback_outputs_1 = detection_model_2(input_tensor)
        outputs = fallback_outputs_1
    else:
        outputs = primary_outputs

    # Attempt with second model if first failed
    if len(outputs[0]["boxes"]) == 0:
        no_detections += 1
        no_detection_images.append(img_name)

        # Perform 5 fixed crops: 4 corners and center
        crop_size = 512  # Assuming you want to use the same size as your detection model input
        crops = [
            image.crop((0, 0, crop_size, crop_size)),  # Top-left
            image.crop((original_width - crop_size, 0, original_width, crop_size)),  # Top-right
            image.crop((0, original_height - crop_size, crop_size, original_height)),  # Bottom-left
            image.crop(
                (original_width - crop_size, original_height - crop_size, original_width, original_height)
            ),  # Bottom-right
            image.crop(
                (
                    (original_width - crop_size) // 2,
                    (original_height - crop_size) // 2,
                    (original_width + crop_size) // 2,
                    (original_height + crop_size) // 2,
                )
            ),  # Center
        ]

        # Save the cropped images
        for i, crop in enumerate(crops):
            cropped_image_path = os.path.join(cropped_images_dir, f"{img_name[:-4]}_crop{i}.jpg")
            crop.save(cropped_image_path)
            detection_scores[cropped_image_path] = 1.0  # Assuming a default high confidence score for manual crops

        continue

    # Crop images based on the bounding boxes
    for output in outputs:
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        for box, score in zip(boxes, scores):
            x1 = int(box[0] * original_width / 512)
            y1 = int(box[1] * original_height / 512)
            x2 = int(box[2] * original_width / 512)
            y2 = int(box[3] * original_height / 512)

            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image_path = os.path.join(cropped_images_dir, f"{img_name[:-4]}_{x1}_{y1}_{x2}_{y2}.jpg")
            cropped_image.save(cropped_image_path)
            detection_scores[cropped_image_path] = score

print("Cropped images saved to:", cropped_images_dir)
print(f"Secondary model used for {secondary_model_used} images.")
print(f"No detections for {no_detections} images.")
print(f"No Detection Image List: {no_detection_images}")


# Image Classification #################################################################################################

prediction_dict = {}
# Process each cropped image for classification
for cropped_image_name in os.listdir(cropped_images_dir):
    img_path = os.path.join(cropped_images_dir, cropped_image_name)
    image = Image.open(img_path).convert("RGB")

    # Transform the image for the model
    transformed_image = classification_transformer(image)
    input_tensor = transformed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = classification_model(input_tensor).squeeze(0).cpu().numpy()

    prediction_dict[cropped_image_name] = predictions

print("Done Making Predictions")

# Post Process the Predictions #####################################################################################
processed_predictions = {}

# Assume the filename is something like "img_1_174_1060_2495_1621.jpg" -or- "img_1.jpg"
for cropped_image_name, prediction in prediction_dict.items():
    # Drop High Background Predictions
    if prediction[-1] > 0.75:
        prediction = prediction[:-1]  # Drop background
        continue

    if np.max(prediction[:-1]) < 0.3:
        continue

    # # Adjust low probabilities to 0
    prediction = prediction[:-1]  # Drop background
    # prediction = [0 if p < 0.3 else p for p in prediction]

    match = re.match(r"(.+?)(?:_\d+_\d+_\d+_\d+)?\.jpg", cropped_image_name)
    if match:
        original_image_name = match.group(1)

    # if original_image_name in processed_predictions:
    #     current_probs = processed_predictions[original_image_name]
    #     combined_probs = [combine_probabilities([current_probs[i], prediction[i]]) for i in range(len(prediction))]
    #     processed_predictions[original_image_name] = combined_probs
    # else:
    #     processed_predictions[original_image_name] = prediction

    if original_image_name in processed_predictions:
        processed_predictions[original_image_name] = np.maximum(processed_predictions[original_image_name], prediction)
    else:
        processed_predictions[original_image_name] = prediction

# Training Data Class Distribution
class_counts = [357, 332, 256, 371, 587, 327, 250, 278, 735, 830, 807, 329, 656, 348, 463]
total_count = sum(class_counts)
# default_probability = [count / total_count for count in class_counts]
default_probability = [1 / 15] * 15

for img in test_images:
    img_name = img.split(".")[0]
    if img_name not in processed_predictions:
        processed_predictions[img_name] = default_probability

print("Done Post Processing Predictions")
count = 0
for key, value in processed_predictions.items():
    if count < 5:
        print(f"{key}: {value}")
        count += 1
    else:
        break

# Submission Creation ##################################################################################################
# Ensure the directory exists
submission_dir = "./submissions"
os.makedirs(submission_dir, exist_ok=True)

# Generate and save the submission file
submission_filename = os.path.join(submission_dir, "submission.csv")
create_submission(processed_predictions, "input/sample_submission.csv", submission_filename)

# Cleanup Cuda Cache ###################################################################################################
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear CUDA cache if available
