import torchvision
import torch
from torch.nn import Module
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path


def get_object_detection_model(num_classes: int, weights: Path | None = None) -> Module:

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if weights:
        
        model.load_state_dict(
            torch.load(weights, map_location=lambda storage, loc: storage)
        )

    return model