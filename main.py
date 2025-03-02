import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to the YOLO weights."
    )
    
    return parser.parse_args()



def main(weights: Path) -> None:

    model = YOLO(weights)
    
    cap = cv2.VideoCapture(1)
    
    while cap.isOpened():
        captured, frame = cap.read()
        
        if captured:
            result: list[Results] = model(frame, show=True)
            
            cv2.imshow("Online tracking", result[0].orig_img)
            
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":

    args = parse_arguments()
    
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted process.")