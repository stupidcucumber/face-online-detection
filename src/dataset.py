import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import Literal
from pathlib import Path


class FaceDetectionDataset(Dataset):
    def __init__(self, root: Path, subset: Literal["train", "val"], width: int, height: int) -> None:
        super(FaceDetectionDataset, self).__init__()
        self.width = width
        self.height = height
        self.root = root
        self.subset = subset
        self.images = list((root / "images" / subset).glob("*.*"))
        
    def __len__(self) -> int:
        return len(self.images)
    
    def _read_image(self, path: Path) -> torch.Tensor:
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA) / 255.0
        return torch.as_tensor(img_res, dtype=torch.float32).permute(2, 0, 1)
    
    def _read_label(self, image: Path, path: Path, image_id: int) -> dict:
        
        image_height, image_width = cv2.imread(image).shape[:2]
        
        classes = []
        bboxes = []
        
        for line in path.read_text().splitlines():
            
            values = line.strip().split(" ")
            
            label = int(values[0]) + 1
            
            center_x_norm, center_y_norm, width_norm, height_norm = [float(value) for value in values[1:]]
            
            width = width_norm * image_width
            height = height_norm * image_height
            xmin = center_x_norm * image_width - width / 2
            ymin = center_y_norm * image_height - height / 2
            xmax = xmin + width
            ymax = ymin + width
            
            bboxes.append(
                [
                    xmin * self.width / image_width, 
                    ymin * self.height / image_height, 
                    xmax * self.width / image_width, 
                    ymax * self.height / image_height
                ]
            )
            classes.append(label)
            
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            
        result = {
            "boxes": boxes,
            "labels": labels,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": iscrowd,
            "image_id": image_id
        }
        
        return result
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        image_path = self.images[index]
        label_path = self.root / "labels" / self.subset / f"{image_path.stem}.txt"
        
        return (
            self._read_image(image_path),
            self._read_label(image_path, label_path, index)
        )