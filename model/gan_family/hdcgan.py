# 학습 안 됨

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        # nz: input noise dimension
        # nc: number of channels, RGB
        # ngf: number of generator filters
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # (b, 100) -> (b, 1024, 4, 4)
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(ngf*8),
            nn.SELU(True),

            # (b, 1024, 4, 4) -> (b, 512, 8, 8)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.SELU(True),

            # (b, 512, 8, 8) -> (b, 256, 16, 16)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.SELU(True),

            # (b, 256, 16, 16) -> (b, 128, 32, 32)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.SELU(True),

            # (b, 128, 32, 32) -> (b, 3, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, input):
        output = self.main(input)

        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        # nc: number of channels
        # ndf: number of discriminator filter
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.SELU(True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.SELU(True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.SELU(True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm2d(ndf*8),
            nn.SELU(True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, input):
        output = self.main(input)
        return output