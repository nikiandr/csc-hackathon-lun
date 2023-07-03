from model import (
    SiameseNetworkDataset,
    ContrastiveLoss,
    SiameseNetwork
)

import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import wandb

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


DATA_PATH = Path("../data")
COMP_DATA_PATH = Path("../data")
IMAGE_PATH = Path("../dataset")

BATCH_SIZE = 64
EPOCHS = 50
IMG_SIZE = (512, 512)

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

train_dataset = SiameseNetworkDataset(train_split_filtered,
                                      transform=train_transforms)
val_dataset = SiameseNetworkDataset(val_split_filtered,
                                    transform=val_transforms)

train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = SiameseNetwork(img_size=IMG_SIZE).to(device)
loss_fn = ContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(train_dataloader),
                        total=len(train_dataloader)):
        # Every data instance is an input + label pair
        input1, input2, labels = data

        # to gpu
        input1 = input1.to(device)
        input2 = input2.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output1, output2 = net(input1, input2)

        # Compute the loss and its gradients
        loss = loss_fn(output1, output2, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % len(train_dataloader) == (len(train_dataloader) - 1):
            last_loss = running_loss / (len(train_dataloader) - 1)
            running_loss = 0.
    wandb.log({"train_loss": last_loss})
    return last_loss


wandb.init(project="csc_hackathon_lun", name="siamese_512_50epochs")
best_vloss = 1_000_000.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for epoch_number in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    avg_loss = train_one_epoch(epoch_number)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    net.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vinput1, vinput2, vlabels = vdata
            # to gpu
            vinput1 = vinput1.to(device)
            vinput2 = vinput2.to(device)
            vlabels = vlabels.to(device)
            voutput1, voutput2 = net(vinput1, vinput2)
            vloss = loss_fn(voutput1, voutput2, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    wandb.log({"val_loss": avg_vloss})
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1
