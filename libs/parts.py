import torch.nn as nn


# Middle layer of Encoder
class Pyramidconv2D(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_feat)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, img):
        img = self.conv(img)
        img = self.bn(img)
        out_img = self.relu(img)
        return out_img


# Middle layer of Decoder
class PyramidconvT2D(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            in_feat, out_feat, kernel_size=4, stride=2, padding=1
        )
        self.bn = nn.BatchNorm2d(out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        img = self.convt(img)
        img = self.bn(img)
        out_img = self.relu(img)
        return out_img


# First layer of Encoder
class Firstconv2D(nn.Module):
    def __init__(self, channel, out_feat):
        super().__init__()
        self.conv = nn.Conv2d(channel, out_feat, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, img):
        img = self.conv(img)
        out_img = self.relu(img)
        return out_img


# First layer of Decoder
class FirstconvT2D(nn.Module):
    def __init__(self, channel, out_feat):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            channel, out_feat, kernel_size=4, stride=1, padding=0
        )
        self.bn = nn.BatchNorm2d(out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        img = self.convt(img)
        img = self.bn(img)
        out_img = self.relu(img)
        return out_img


# Final layer of Decoder
class FinalconvT2D(nn.Module):
    def __init__(self, in_feat, channel):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            in_feat, channel, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, img):
        img = self.convt(img)
        out_img = self.tanh(img)
        return out_img


class Encoder(nn.Module):
    def __init__(self, z_dim, channel):
        super().__init__()
        self.firstconv = Firstconv2D(channel, 64)
        self.pyramidconv1 = Pyramidconv2D(64, 128)
        self.pyramidconv2 = Pyramidconv2D(128, 256)
        self.pyramidconv3 = Pyramidconv2D(256, 512)
        self.finalconv = nn.Conv2d(512, z_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, img):
        img = self.firstconv(img)
        img = self.pyramidconv1(img)
        img = self.pyramidconv2(img)
        img = self.pyramidconv3(img)
        out_img = self.finalconv(img)
        return out_img, img


class Decoder(nn.Module):
    def __init__(self, z_dim, channel):
        super().__init__()
        self.firstconvt = FirstconvT2D(z_dim, 512)
        self.pyramidconvt1 = PyramidconvT2D(512, 256)
        self.pyramidconvt2 = PyramidconvT2D(256, 128)
        self.pyramidconvt3 = PyramidconvT2D(128, 64)
        # self.finalconvt = FinalconvT2D(64, channel)
        self.finalconvt = nn.ConvTranspose2d(
            64, channel, kernel_size=4, stride=2, padding=1
        )

    def forward(self, img):
        img = self.firstconvt(img)
        img = self.pyramidconvt1(img)
        img = self.pyramidconvt2(img)
        img = self.pyramidconvt3(img)
        out_img = self.finalconvt(img)
        return out_img
