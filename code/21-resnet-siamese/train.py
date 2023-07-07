from model import (
    SiameseNetworkDataset,
    SiameseResNet
)

# import wandb
import argparse
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
# import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='Train siamese network.')
parser.add_argument('--wandb', dest='wandb',
                    default="", type=str,
                    help='run name (default: "")')
parser.add_argument('--batch', dest='batch_size',
                    default=4, type=int,
                    help='batch size (default: 4)')
parser.add_argument('--epochs', dest='epochs',
                    default=5, type=int,
                    help='number of epochs (default: 5)')
parser.add_argument('--width', dest='width',
                    default=1024, type=int,
                    help='width of input images (default: 512)')
parser.add_argument('--height', dest='height',
                    default=1024, type=int,
                    help='height of input images (default: 512)')
parser.add_argument('--loader_workers', dest='num_workers',
                    default=20, type=int,
                    help='number of workers for dataloaders (default: 20)')

args = parser.parse_args()

DATA_PATH = Path("../../data")
COMP_DATA_PATH = Path("../../data")
IMAGE_PATH = Path("../../dataset")

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
IMG_SIZE = (args.width, args.height)

TRAIN_SPLIT_FILTERED_PATH = DATA_PATH / "train_split_filtered.csv"
VAL_SPLIT_FILTERED_PATH = DATA_PATH / "val_split_filtered.csv"
train_split_filtered = pd.read_csv(TRAIN_SPLIT_FILTERED_PATH)
val_split_filtered = pd.read_csv(VAL_SPLIT_FILTERED_PATH)
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE)),
        transforms.ToTensor()
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE)),
        transforms.ToTensor()
    ]
)
train_dataset = SiameseNetworkDataset(train_split_filtered[:20],
                                      transform=train_transforms)
val_dataset = SiameseNetworkDataset(val_split_filtered[:20],
                                    transform=val_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=args.num_workers)


if args.wandb == "":
    wandb_logger = WandbLogger(project="csc_hackathon_lun")
else:
    wandb_logger = WandbLogger(name=args.wandb,
                               project="csc_hackathon_lun")


model = SiameseResNet()

trainer = pl.Trainer(
    limit_train_batches=100,
    max_epochs=100,
    log_every_n_steps=50,
    accelerator='cpu',
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    logger=wandb_logger)

trainer.fit(model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)
