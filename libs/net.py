import torch.nn as nn

from .parts import Encoder, Decoder

"""
Network of Reconstructor and Discriminator
Expected Input : (Batch, channel, 64, 64)
"""


class Reconstructor(nn.Module):
    def __init__(self, z_dim, channel):
        super().__init__()
        self.encoder1 = Encoder(z_dim, channel)
        self.decoder = Decoder(z_dim, channel)
        self.encoder2 = Encoder(z_dim, channel)

    def forward(self, img):
        feat_i, _ = self.encoder1(img)
        fakeimg = self.decoder(feat_i)
        feat_o, _ = self.encoder2(fakeimg)
        return fakeimg, feat_i, feat_o


class Discriminator(nn.Module):
    def __init__(self, z_dim, channel):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim, channel)
        self.discriminator = nn.Conv2d(512, 1, kernel_size=4, stride=4, padding=0)

    def forward(self, img):
        _, feature = self.encoder(img)
        pred = self.discriminator(feature).squeeze()
        return pred, feature


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.uniform_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
