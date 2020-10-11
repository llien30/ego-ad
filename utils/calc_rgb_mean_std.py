import torch

from libs.dataset import DatasetRGB
from torch.utils.data import DataLoader

import argparse
import yaml

from addict import Dict


def get_parameters():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(description="take config file path")

    parser.add_argument("config", type=str, help="path of a config file for training")

    parser.add_argument("--no_wandb", action="store_true", help="Add --no_wandb option")

    return parser.parse_args()


args = get_parameters()

CONFIG = Dict(yaml.safe_load(open(args.config)))

train_dataset = DatasetRGB(
    csv_file=CONFIG.train_csv_file,
    datasetdir=CONFIG.datasetdir,
    input_size=CONFIG.input_size,
    transform=None,
)

loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
print(len(loader.dataset))
mean = 0
std = 0

for sample in loader:
    image = sample["img"]
    image = image.type(torch.DoubleTensor)
    batch_samples = image.size(0)
    image = image.view(batch_samples, image.size(1), -1)
    # image = image / 255
    mean += image.mean(2).sum(0)
mean = mean / len(loader.dataset)

var = 0.0
for sample in loader:
    image = sample["img"]
    image = image.type(torch.DoubleTensor)
    batch_samples = image.size(0)
    image = image.view(batch_samples, image.size(1), -1)
    # image = image / 255
    var += ((image - mean.unsqueeze(1)) ** 2).sum([0, 2])
std = torch.sqrt(var / (len(loader.dataset) * 224 * 224))


print(mean)
print(std)
