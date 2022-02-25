"""
For Internal Use Only.
Generator and Discriminator Networks for CycleGAN.
"""
import torch
import torch.nn as nn


# from original paper
# Architecture of Generator
# The network with 6 residual blocks consists of:
# c7s1-64,d128,d256,R256,R256,R256,
# R256,R256,R256,u128,u64,c7s1-3
# The network with 9 residual blocks consists of:
# c7s1-64,d128,d256,R256,R256,R256,
# R256,R256,R256,R256,R256,R256,u128
# u64,c7s1-3

class CommonConvolutionBlock(nn.Module):
    """
    Common Convolution Block for Usage.
    Includes:
        1. Convolution Layer
        2. InstanceNorm Layer
        3. ReLU Activation
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            filters,
            down_sample=True,
            **kwargs
    ):
        """

        Parameters
        ----------
        in_channels: Input channels
        out_channels: Output Channels
        filters: Filter Size
        padding: Padding in Conv, leave if want to use default
        stride: Stride in Conv Block
        """
        super(CommonConvolutionBlock, self).__init__()
        if down_sample:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, filters, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, filters, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.model(x)


# Residual Block - Either to be used 6 times or 9 times in final generater
class ResidualBlock(nn.Module):
    """
    Residual Block for Generator Network.
    Includes 2 Layers of:
    1. Reflection Padding to reduce Artifacts
    2. Conv2d
    3. InstanceNorm
    4. ReLU
    """

    def __init__(self, in_channels):
        """

        Parameters
        ----------
        in_channels: Input channels for Block, output channels are equal.
        """
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            CommonConvolutionBlock(in_channels, in_channels, 3),
            nn.ReflectionPad2d(1),
            CommonConvolutionBlock(in_channels, in_channels, 3)
        )

    def forward(self, x):
        return x + self.model(x)


# Generator Network Implementation
class Generator(nn.Module):
    """
    Generator Model for CycleGAN.
    """

    def __init__(
            self,
            in_channels,
            residual_blocks=6,
            **kwargs
    ):
        """

        Parameters
        ----------
        in_channels: Input channels
        residual_blocks: No of Residual Blocks
        kwargs: Other keyword arguments
        """
        super(Generator, self).__init__()

        layers = []
        out_channels = 64
        # initial layer c7s1-64
        layers += [
            nn.ReflectionPad2d(in_channels),
            CommonConvolutionBlock(in_channels, out_channels, 7),
        ]

        # 2 dk layers
        layers += [
            CommonConvolutionBlock(out_channels, out_channels * 2, 3, stride=2, padding=2),
            CommonConvolutionBlock(out_channels * 2, out_channels * 4, 3, stride=2, padding=2),
        ]

        out_channels = out_channels * 4

        # Residual Blocks
        for _ in range(residual_blocks):
            layers += [ResidualBlock(out_channels)]

        # Fractional Stride layers - TransConv
        layers += [
            CommonConvolutionBlock(out_channels, out_channels // 2, 3, down_sample=False, stride=2, padding=1,
                                   output_padding=1),
            CommonConvolutionBlock(out_channels // 2, out_channels // 4, 3, down_sample=False, stride=2, padding=1,
                                   output_padding=1),
        ]

        out_channels //= 4

        layers += [
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(out_channels, in_channels, 7)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.model(x))


# Architecture for Discriminator
"""we use 70 × 70 PatchGAN [22]. Let Ck denote a
4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k
filters and stride 2. After the last layer, we apply a 
convolution to produce a 1-dimensional output. We do not use
InstanceNorm for the first C64 layer. We use leaky ReLUs
with a slope of 0.2. The discriminator architecture is:
C64-C128-C256-C512"""


class Discriminator(nn.Module):
    """
    Discriminator for CycleGAN.
    """

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        layers = []

        # first layer
        layers += [
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        ]

        out_channels = 64
        # next 3 layers
        for _ in range(3):
            layers += [
                nn.Conv2d(out_channels, out_channels * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels * 2),
                nn.LeakyReLU(0.2)
            ]
            out_channels *= 2

        # one last layer
        layers += [nn.ZeroPad2d((1, 0, 1, 0))]
        layers += [nn.Conv2d(out_channels, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def test_discriminator():
    img_channels = 1
    dims = 128
    sample_img = torch.randn((1, img_channels, dims, dims))

    gen = Discriminator(1)

    output = gen(sample_img)
    print(gen)

    print(output.shape)


def test_generator():
    img_channels = 1
    dims = 128
    sample_img = torch.randn((1, img_channels, dims, dims))

    gen = Generator(img_channels)

    output = gen(sample_img)

    print(output.shape)


if __name__ == '__main__':
    test_generator()


