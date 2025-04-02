from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Literal
from src.utils.engine import evaluate, train_one_epoch


def train(
    model: Module,
    optimizer: Optimizer,
    epochs: int,
    lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: Literal["cpu", "cuda"]
) -> None:

    for epoch in range(epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_dataloader, device=device)