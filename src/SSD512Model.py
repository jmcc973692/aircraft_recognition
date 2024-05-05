import torch
from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator, _vgg_extractor
from torchvision.models.vgg import VGG19_BN_Weights, vgg19_bn
from torchvision.ops import nms


class SSD512Model(torch.nn.Module):
    def __init__(self, num_classes, detection_threshold=0.2, iou_threshold=0.3, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.detection_threshold = detection_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(num_classes)

    def _load_model(self, num_classes):
        backbone = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        backbone = _vgg_extractor(backbone=backbone, highres=True, trainable_layers=0)
        size = (512, 512)
        # Default Box configuration for SSD512
        # anchor_generator = DefaultBoxGenerator(
        #     [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        #     scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.15],
        #     steps=[4, 8, 16, 32, 64, 100, 300],
        # )
        # Default Box Configuration Calculated from Data
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=[
                [0.8971957329869933],
                [1.5642842606124487],
                [2.2471642914741086],
                [2.95087410647832],
                [3.7204049800429164],
                [4.848641946479785],
                [7.877610493251331],
            ],
            min_ratio=0.00543145,
            max_ratio=0.9998303,
            scales=[
                0.005431454501732846,
                0.171164590936979,
                0.3368977273722251,
                0.5026308638074714,
                0.6683640002427175,
                0.8340971366779636,
                0.9998302731132097,
                1.1,
            ],
        )
        image_mean = [0.48236, 0.45882, 0.40784]
        image_std = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        model = SSD(
            backbone=backbone,
            anchor_generator=anchor_generator,
            size=size,
            num_classes=num_classes,
            image_mean=image_mean,
            image_std=image_std,
            score_thresh=self.detection_threshold,
            iou_thresh=self.iou_threshold,
            detections_per_img=50,
        )

        return model.to(self.device)

    def forward(self, images, targets=None):
        if targets is not None:
            outputs = self.model(images, targets)
            return outputs

        outputs = self.model(images)

        # filtered_outputs = []
        # for output in outputs:
        #     boxes = output["boxes"]
        #     scores = output["scores"]
        #     labels = output["labels"]

        #     high_confidence_idxs = scores > self.detection_threshold
        #     boxes = boxes[high_confidence_idxs]
        #     scores = scores[high_confidence_idxs]
        #     labels = labels[high_confidence_idxs]

        #     keep_idxs = nms(boxes, scores, self.iou_threshold)

        #     filtered_outputs.append(
        #         {"boxes": boxes[keep_idxs], "scores": scores[keep_idxs], "labels": labels[keep_idxs]}
        #     )

        return outputs
