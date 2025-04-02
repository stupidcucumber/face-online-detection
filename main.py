import cv2
from typing import Any
from ultralytics import YOLO
from src.face_rcnn import get_object_detection_model
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
import numpy as np
from typing import Literal


def parse_video_capture_address(arg: Any) -> int | str:
    _result = str(arg)
    if _result.isnumeric():
        return int(_result)
    return _result


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument(
        "--model", type=str, default="yolo", help="Which model to use, by default it is yolo. Possible models: faster-rcnn, yolo"
    )
    
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to the YOLO weights."
    )
    
    parser.add_argument(
        "--video-capture-address", type=parse_video_capture_address, default=0, help="Address of the camera/video to capture from." 
    )
    
    return parser.parse_args()


def predict_faster_rcnn(
    frame: cv2.Mat,
    detector: torch.nn.Module,
    width: int = 512,
    height: int = 512
) -> None:
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = cv2.resize(img_rgb, (width, height), cv2.INTER_AREA) / 255.0
    image = torch.as_tensor(img_res, dtype=torch.float32).permute(2, 0, 1)
    
    predictions = detector(image)
    
    print(predictions)


def main(model: Literal["yolo", "faster-rcnn"], weights: Path, video_capture_address: str | int) -> None:

    if model == "yolo":
        
        detector = YOLO(weights)
        
    else:
        
        detector = get_object_detection_model(2, weights)
    
    cap = cv2.VideoCapture(video_capture_address)
    
    while cap.isOpened():
        captured, frame = cap.read()
        
        if captured:
            
            if model == "yolo":
            
                detector(frame, show=True)
            
            else:
                
                predictions = ...


if __name__ == "__main__":

    args = parse_arguments()
    
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted process.")