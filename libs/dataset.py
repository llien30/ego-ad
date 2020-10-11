from torch.utils import data
import pandas as pd
import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image


class Dataset2D(data.Dataset):
    def __init__(self, csv_file, datasetdir, input_size, transform="None"):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.datasetdir = datasetdir
        self.input_size = input_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["img_path"]
        img = np.load(img_path)
        # img = np.fromfile(img_path,np.float32)
        img = np.reshape(img, (224, 224, 2))
        img = np.transpose(img, (2, 0, 1))
        # print(img.shape)

        if self.transform is not None:
            img = torch.tensor(img)
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(self.input_size, self.input_size))
            img = img.squeeze(0)
            img = self.transform(img)

        cls_id = self.df.iloc[idx]["cls_id"]
        cls_label = self.df.iloc[idx]["cls_label"]

        sample = {
            "img": img,
            "cls_id": cls_id,
            "label": cls_label,
            "img_path": img_path,
        }

        return sample


class DatasetRGB(data.Dataset):
    def __init__(self, csv_file, datasetdir, input_size, transform="None"):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.datasetdir = datasetdir
        self.input_size = input_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["img_path"]
        img = Image.open(img_path)
        # img = np.fromfile(img_path,np.float32)
        img = np.array(img)
        img = np.reshape(img, (224, 224, 3))
        img = np.transpose(img, (2, 0, 1))
        # print(img.shape)

        if self.transform is not None:
            img = torch.tensor(img)
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(self.input_size, self.input_size))
            img = img.squeeze(0)
            img = self.transform(img)

        cls_id = self.df.iloc[idx]["cls_id"]
        cls_label = self.df.iloc[idx]["cls_label"]

        sample = {
            "img": img,
            "cls_id": cls_id,
            "label": cls_label,
            "img_path": img_path,
        }

        return sample
