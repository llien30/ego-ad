import torch
from torch.utils.data import DataLoader

from libs.net import Reconstructor, Discriminator, weights_init
from libs.model import egocentric_ad
from libs.dataset import Dataset2D
from libs.transform import ImageTransform
from libs.statistics import get_mean_std

from addict import Dict
import argparse
import yaml
import os
import wandb


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


if not args.no_wandb:
    wandb.init(
        config=CONFIG,
        name=CONFIG.name,
        project="egocentric_2D",  # have to change when you want to change project
        # jobtype="training",
    )

mean, std = get_mean_std()
train_dataset = Dataset2D(
    CONFIG.train_csv_file,
    CONFIG.datasetdir,
    CONFIG.input_size,
    transform=ImageTransform(mean, std),
)
test_dataset = Dataset2D(
    CONFIG.test_csv_file,
    CONFIG.datasetdir,
    CONFIG.input_size,
    transform=ImageTransform(mean, std),
)

train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=CONFIG.test_batch_size, shuffle=False
)


R = Reconstructor(CONFIG.z_dim, CONFIG.channel)
D = Discriminator(CONFIG.z_dim, CONFIG.channel)

R.apply(weights_init)
D.apply(weights_init)

if not args.no_wandb:
    # Magic
    wandb.watch(R, log="all")
    wandb.watch(D, log="all")

R_update, D_update = egocentric_ad(
    R,
    D,
    CONFIG.num_epochs,
    CONFIG.z_dim,
    CONFIG.lambdas,
    dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    no_wandb=args.no_wandb,
)
if not os.path.exists(CONFIG.save_dir):
    os.makedirs(CONFIG.save_dir)

torch.save(
    R_update.state_dict(),
    os.path.join(CONFIG.save_dir, "R-{}.prm".format(CONFIG.name)),
)
torch.save(
    D_update.state_dict(),
    os.path.join(CONFIG.save_dir, "D-{}.prm".format(CONFIG.name)),
)

print("Done")
