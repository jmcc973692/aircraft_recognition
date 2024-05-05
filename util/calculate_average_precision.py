import torch
from torchvision.ops import box_iou


def calculate_average_precision(detections, ground_truths):
    """
    Calculate average precision for a single image.

    Args:
        detections (list of dict): Each dictionary contains 'boxes', 'scores', and 'labels'.
        ground_truths (dict): Contains 'boxes' and 'labels'.

    Returns:
        float: The average precision for the detections against the ground truths.
    """
    # Convert ground truth boxes and labels
    gt_boxes = ground_truths["boxes"]
    # gt_labels = ground_truths["labels"]

    # Initialize containers for true positives and scores
    all_scores = []
    matched_gt_boxes = torch.zeros(gt_boxes.size(0), dtype=torch.bool)

    for detection in detections:
        boxes = detection["boxes"]
        scores = detection["scores"]
        # labels = detection["labels"]

        # Append scores
        all_scores.extend(scores.tolist())

        # Calculate IoU between detection boxes and ground truth boxes
        ious = box_iou(boxes, gt_boxes)

        # Match detections to ground truth boxes
        max_iou, max_indices = ious.max(dim=1)
        is_positive = max_iou >= 0.5

        # Mark ground truth boxes as matched
        for i, idx in enumerate(max_indices[is_positive]):
            if not matched_gt_boxes[idx]:
                matched_gt_boxes[idx] = True
            else:
                is_positive[i] = False  # This detection is a duplicate and counts as a false positive

    # Calculate true positives and false positives
    tp = matched_gt_boxes.sum().item()
    fp = len(detections) - tp
    fn = len(gt_boxes) - tp

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate average precision
    ap = (precision + recall) / 2 if precision + recall > 0 else 0

    return ap


def test_average_precision():
    # Perfect match
    detections_1 = [
        {
            "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.7, 0.7, 0.9, 0.9]]),
            "scores": torch.tensor([0.9, 0.85]),
            "labels": torch.tensor([1, 1]),
        }
    ]
    ground_truths_1 = {
        "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.7, 0.7, 0.9, 0.9]]),
        "labels": torch.tensor([1, 1]),
    }

    # Partial match
    detections_2 = [
        {
            "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.6, 0.6]]),
            "scores": torch.tensor([0.9, 0.4]),
            "labels": torch.tensor([1, 1]),
        }
    ]
    ground_truths_2 = {
        "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.7, 0.7, 0.9, 0.9]]),
        "labels": torch.tensor([1, 1]),
    }

    ap1 = calculate_average_precision(detections_1, ground_truths_1)
    ap2 = calculate_average_precision(detections_2, ground_truths_2)

    print("Average Precision Test 1 (Perfect match):", ap1)
    print("Average Precision Test 2 (Partial match):", ap2)


if __name__ == "__main__":
    test_average_precision()
