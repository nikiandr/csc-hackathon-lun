from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from PIL import ImageFile
import pandas as pd


ImageFile.LOAD_TRUNCATED_IMAGES = True


class SiameseNetworkDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, transform=None):
        self.transform = transform
        self.split_df = split_df

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, index):
        row = self.split_df.iloc[index]
        img1 = Image.open(row["image_path1"])
        img2 = Image.open(row["image_path2"])
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


class SiameseNetwork(nn.Module):
    def __init__(self, img_size=(512, 512)):
        super().__init__()
        self.img_size = img_size
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # output (16, IMG_SIZE[0] // 2, IMG_SIZE[1] // 2)
            # nn.Dropout2d(p=.2),

            nn.Conv2d(16, 32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # output (32, IMG_SIZE[0] // 4, IMG_SIZE[1] // 4)
            # nn.Dropout2d(p=.2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # output (64, IMG_SIZE[0] // 8, IMG_SIZE[1] // 8)
            # nn.Dropout2d(p=.2),

            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # output (128, IMG_SIZE[0] // 16,
                              # IMG_SIZE[1] // 16)
            # nn.Dropout2d(p=.2),

            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # output (128, IMG_SIZE[0] // 32,
                              # IMG_SIZE[1] // 32)
            # nn.Dropout2d(p=.2),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.img_size[0] // 32) *
                      (self.img_size[1] // 32),
                      1024),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x1, x2):
        x1 = self.cnn(x1)
        x1 = self.flatten(x1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)

        x2 = self.cnn(x2)
        x2 = self.flatten(x2)
        x2 = self.fc1(x2)
        x2 = self.fc2(x2)
        x2 = self.fc3(x2)
        return x1, x2
