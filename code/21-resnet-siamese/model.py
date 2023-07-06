import torch
import torchvision
import torch.nn.functional as F
from torch import optim, nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score
import numpy as np
from PIL import Image
from PIL import ImageFile
import pandas as pd
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


class SiameseNetworkDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, transform=None):
        self.transform = transform
        self.split_df = split_df

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, index):
        row = self.split_df.iloc[index]
        img1 = Image.open(row["image_path1"]).convert('RGB')
        img2 = Image.open(row["image_path2"]).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, row["is_same"]


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) *
                                      torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
            ))
        return loss_contrastive


# define the LightningModule
class SiameseResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet34()
        self.fc = nn.Linear(1000, 1)
        self.loss = ContrastiveLoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x1, x2, y = batch
        output1, output2 = self.resnet(x1), self.resnet(x2)
        output1, output2 = self.fc(output1), self.fc(output2)
        loss = self.loss(output1, output2, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        output1, output2 = self.resnet(x1), self.resnet(x2)
        output1, output2 = self.fc(output1), self.fc(output2)
        loss = self.loss(output1, output2, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        dist = torch.abs(output1 - output2)
        res = {}
        for i in np.arange(torch.min(dist).item(), torch.max(dist).item()):
            res[i] = f1_score(y, (dist < i).long(),
                              labels=[0, 1],
                              zero_division=0)
        threshold = max(res, key=res.get)
        self.log("val_best_threshold", threshold)
        self.log("val_best_f1", res[threshold])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
