import cv2
from typing import Any
from ultralytics import YOLO
from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_video_capture_address(arg: Any) -> int | str:
    _result = str(arg)
    if _result.isnumeric():
        return int(_result)
    return _result


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to the YOLO weights."
    )
    
    parser.add_argument(
        "--video-capture-address", type=parse_video_capture_address, default=0, help="Address of the camera/video to capture from." 
    )
    
    return parser.parse_args()



def main(weights: Path, video_capture_address: str | int) -> None:

    model = YOLO(weights)
    
    cap = cv2.VideoCapture(video_capture_address)
    
    while cap.isOpened():
        captured, frame = cap.read()
        
        if captured:
            model(frame, show=True)


if __name__ == "__main__":

    args = parse_arguments()
    
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted process.")