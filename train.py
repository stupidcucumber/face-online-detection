from src.training import train
from src.face_rcnn import get_object_detection_model
from src.utils.utils import collate_fn
from torch.utils.data import DataLoader
from src.dataset import FaceDetectionDataset
import argparse
import torch
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--root", type=Path, required=True, help="Path to the data root of the dataset."
    )
    
    parser.add_argument(
        "-b", type=int, default=12, help="Batch size number. By default it is 12."
    )
    
    parser.add_argument(
        "-e", type=int, default=10, help="Number of epochs to train for."
    )
    
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to perform training."
    )
    
    return parser.parse_args()


def main(root: Path, b: int, e: int, device) -> None:
    
    IMAGE_WIDTH = 512
    IMAGE_HEIGH = 512

    train_dataset = FaceDetectionDataset(
        root=root,
        subset="train",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGH
    )
    
    val_dataset = FaceDetectionDataset(
        root=root,
        subset="val",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGH
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=b, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=b,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = get_object_detection_model(2)

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=0.005,
        momentum=0.9, 
        weight_decay=0.0005
    )

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    train(
        model,
        optimizer,
        e,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        device=torch.device(device)
    )
    
    torch.save(
        model.state_dict(),
        f"faster-rcnn_{b}_{e}.pt"
    )


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("User interrupted the process.")
