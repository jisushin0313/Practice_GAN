import torch
import torch.nn as nn


def upsample_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2),

        nn.ConvTranspose2d(in_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2),
    )


def downsample_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        nn.BatchNorm2d(in_channel),
        nn.LeakyReLU(0.2),

        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2),

        nn.AvgPool2d(kernel_size=2),
    )


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        # nz: input noise dimension
        # nc: number of channels, RGB
        # ngf: number of generator filters
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            # (b, 256) -> (b, 512, 4, 4)
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(ngf*8, ngf*8, 3, 1, 1),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2),
        )

        # (b, 512, 4, 4) -> (b, 256, 8, 8)
        self.layer2 = upsample_block(ngf*8, ngf*4)
        # (b, 256, 8, 8) -> (b, 128, 16, 16)
        self.layer3 = upsample_block(ngf*4, ngf*2)
        # (b, 128, 16, 16) -> (b, 64, 32, 32)
        self.layer4 = upsample_block(ngf*2, ngf)
        # (b, 64, 32, 32) -> (b, 3, 64, 64)
        self.layer5 = upsample_block(ngf, nc)
        self.linear_act = nn.Tanh()


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.linear_act(output)

        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        # nc: number of channels
        # ndf: number of discriminator filter
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            # (b, 3, 64, 64) -> (b, 64, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

        # (b, 64, 32, 32) -> (b, 128, 16, 16)
        self.layer2 = downsample_block(ndf, ndf*2)
        # (b, 128, 16, 16) -> (b, 256, 8, 8)
        self.layer3 = downsample_block(ndf*2, ndf*4)
        # (b, 256, 8, 8) -> (b, 512, 4, 4)
        self.layer4 = downsample_block(ndf*4, ndf*8)

        # (b, 512, 4, 4) -> (b, 1, 1, 1)
        self.layer5 = nn.Sequential(
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
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)

        return output