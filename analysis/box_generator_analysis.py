import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans


def load_and_analyze_data(csv_file, img_dir):
    # Load annotations
    df = pd.read_csv(csv_file)

    # Initialize columns for image dimensions
    df["img_width"] = 0
    df["img_height"] = 0

    # Read image sizes and bounding box details
    for index, row in df.iterrows():
        img_path = os.path.join(img_dir, row["filename"] + ".jpg")
        with Image.open(img_path) as img:
            df.at[index, "img_width"], df.at[index, "img_height"] = img.size

    # Calculate bounding box width, height, and aspect ratio
    df["bbox_width"] = df["xmax"] - df["xmin"]
    df["bbox_height"] = df["ymax"] - df["ymin"]
    df["aspect_ratio"] = df["bbox_width"] / df["bbox_height"]
    df["normalized_area"] = (df["bbox_width"] * df["bbox_height"]) / (df["img_width"] * df["img_height"])

    return df


def suggest_default_box_params(df, num_feature_maps=6):
    # Cluster aspect ratios to find the most common ones
    aspect_ratios = df["aspect_ratio"].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans.fit(aspect_ratios)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())  # Sort the cluster centers to maintain consistency

    # Determine min and max ratios based on normalized areas
    min_ratio = np.sqrt(df["normalized_area"].min())
    max_ratio = np.sqrt(df["normalized_area"].max())

    # Calculate scales
    scales = np.linspace(
        min_ratio, max_ratio, num_feature_maps + 1
    ).tolist()  # +1 to include both min and max in the scales

    return {"aspect_ratios": cluster_centers.tolist(), "min_ratio": min_ratio, "max_ratio": max_ratio, "scales": scales}


# Paths
csv_file_path = "data/all_dataset_bounding_boxes.csv"
image_directory = "data/image_dataset/"

# Load data and analyze bounding boxes and image sizes
bbox_data = load_and_analyze_data(csv_file_path, image_directory)

# Calculate suggested parameters for DefaultBoxGenerator
params = suggest_default_box_params(bbox_data)

print("Aspect Ratios:", params["aspect_ratios"])
print("Min Ratio:", params["min_ratio"])
print("Max Ratio:", params["max_ratio"])
print("Scales:", params["scales"])
