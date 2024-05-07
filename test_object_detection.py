import os
import random

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import normalize, to_tensor

from src.ImageTransformer import ImageTransformer
from src.SSD512Model import SSD512Model
from src.SSDLiteModel import SSDLiteModel

# Path setup
model_path = "models/detection_model_SSD512_2024-05-06_14-34-52.pth"
test_images_dir = "input/test_images"  # Directory containing test images
output_dir = "temp/annotated_images"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSD512Model(num_classes=2, detection_threshold=0.2, iou_threshold=0.5, device=device)
# model = SSDLiteModel(num_classes=2, detection_threshold=0.2, iou_threshold=0.3, device=device)
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

# Select random test images
test_images = random.sample(os.listdir(test_images_dir), 20)  # Adjust number of images as needed

transformer = ImageTransformer(
    resize_dims=(512, 512), mean=[0.48236, 0.45882, 0.40784], std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
)

# Processing images
for img_name in test_images:
    img_path = os.path.join(test_images_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    # Get original dimensions
    original_width, original_height = image.size

    transformed_image = transformer(image)

    input_tensor = transformed_image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)

    # Draw bounding boxes on the image using outputs
    draw = ImageDraw.Draw(image)
    for output in outputs:
        boxes = output["boxes"].cpu().numpy()
        for box in boxes:
            # Reverse scale the bounding box coordinates
            x1 = box[0] * original_width / 512
            y1 = box[1] * original_height / 512
            x2 = box[2] * original_width / 512
            y2 = box[3] * original_height / 512
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

    # Save the annotated image
    output_image_path = os.path.join(output_dir, img_name)
    image.save(output_image_path)

print("Annotated images saved to:", output_dir)
