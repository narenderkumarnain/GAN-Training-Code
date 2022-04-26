from PIL import Image
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



import logging
from torchvision.transforms import transforms

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

class CycleGANConfig():
    """
    Variables and default Values are defined here.
    """

    # device
    # options : ["cpu","gpu","tpu"]
    device = "cpu"

    # related to input images
    input_dims = (3,256,256)

    # related to inference requirements
    inference = False
    input_A = True


    # no of residual blocks
    # should be 6 in low resolution training
    # and 9 in high resolution training
    residual_blocks = 6

    # model initial weights
    weights_gen_AB = None
    weights_gen_BA = None
    weights_dis_A = None
    weights_dis_B = None

    # checkpointing
    checkpoint_dir = "model_weights"
    checkpoint_freq = 5

    # sample inference results
    sample_results_dir = "sample_results"
    batch_freq = 100

    # dataset information
    dataset_path = "./dataset/archive/apple2orange/apple2orange"
    aligned = False
    sub_dir_dataset = False

    # training configurations
    batch_size = 1
    n_epochs = 1
    log_freq = 10

    # learning rates
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    decay_epoch = 100

    lambda_cyclic = 10.0
    lambda_identity = 5.0

    def __str__(self):
        """
        String representation for Mode Config
        Returns
        -------

        """
        # TODO Completion of String Representation
        return f'''
        Model Configuration:
            Device: {self.device}
            Input Shape: {self.input_dims}
            Mode: {"Inference" if self.inference else "Training"}
            Generators:
                Residual blocks: {self.residual_blocks}
                Weights A to B: {self.weights_gen_AB if self.weights_gen_AB is not None else "New"}
                Weights B to A: {self.weights_gen_BA if self.weights_gen_BA is not None else "New"}
            Discriminator:
                Weights A: {self.weights_dis_A if self.weights_dis_A is not None else "New"}
                Weights B: {self.weights_dis_B if self.weights_dis_B is not None else "New"}
        '''










"""
For External Use.
Wrapper Model class for CycleGAN.
"""
import os.path
#import wandb
import logging
import torch
import time
import datetime
import tqdm
import torch.nn as nn
import itertools
from torchvision.transforms import transforms
import numpy as np
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader


# some configurations for logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class CycleGAN:
    """
    Class to encapsulate CycleGAN Working.
    Config Class Parameters controls various Functionalities Here.
    """

    def __init__(self, config: CycleGANConfig, project_name = "CycleGAN 0"):
        """

        Parameters
        ----------
        config: Configuration for Model.
        """
        self.config = config
        self.transform = [
            transforms.Resize((self.config.input_dims[1], self.config.input_dims[2])),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        if self.config.input_dims[0] != 1:
            self.transform += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
       # wandb.init(project = project_name)
        if not self.config.inference:
            # two generators
            self.Gen_A_to_B = Generator(self.config.input_dims[0],self.config.residual_blocks)
            self.Gen_B_to_A = Generator(self.config.input_dims[0], self.config.residual_blocks)
            # two discriminators
            self.Dis_A = Discriminator(self.config.input_dims[0])
            self.Dis_B = Discriminator(self.config.input_dims[0])

           # wandb.watch((self.Gen_A_to_B, self.Gen_B_to_A, self.Dis_A, self.Dis_B), log = "all", log_freq = config.log_freq)

            # loading weights
            if self.config.weights_gen_AB is not None:
                self.Gen_A_to_B.load_state_dict(torch.load(self.config.weights_gen_AB,map_location=torch.device('cpu')))
                logger.info(f"Gen_A_to_B: weights loaded from {self.config.weights_gen_AB}")
            if self.config.weights_gen_BA is not None:
                self.Gen_B_to_A.load_state_dict(torch.load(self.config.weights_gen_BA,map_location=torch.device('cpu')))
                logger.info(f"Gen_B_to_A: weights loaded from {self.config.weights_gen_BA}")
            if self.config.weights_dis_A is not None:
                self.Dis_A.load_state_dict(torch.load(self.config.weights_dis_A,map_location=torch.device('cpu')))
                logger.info(f"Dis A: weights loaded from {self.config.weights_dis_A}")
            if self.config.weights_dis_B is not None:
                self.Dis_B.load_state_dict(torch.load(self.config.weights_dis_B,map_location=torch.device('cpu')))
                logger.info(f"Dis B: weights loaded from {self.config.weights_dis_B}")
        else:
            if self.config.input_A:
                self.Gen_A_to_B = Generator(self.config.input_dims[0], self.config.residual_blocks)
                if self.config.weights_gen_AB is not None:
                    self.Gen_A_to_B.load_state_dict(torch.load(self.config.weights_gen_AB))
                    logger.info(f"Gen_A_to_B: weights loaded from {self.config.weights_gen_AB}")
            else:
                self.Gen_B_to_A = Generator(self.config.input_dims[0], self.config.residual_blocks)
                if self.config.weights_gen_BA is not None:
                    self.Gen_B_to_A.load_state_dict(torch.load(self.config.weights_gen_BA))
                    logger.info(f"Gen_B_to_A: weights loaded from {self.config.weights_gen_BA}")


    def predict(self,img, A_to_B = True):
        """

        Parameters
        ----------
        img: Input Image

        Returns
        -------

        """
        transform = transforms.Compose(self.transform)
        img = transform(img).unsqueeze(0)
        if A_to_B:
            self.Gen_A_to_B = self.Gen_A_to_B.to(torch.device(self.config.device))
            res = self.Gen_A_to_B(img)
        else:
            res = self.Gen_B_to_A(img)

        return res



#image = Image.open('ct15.png')
  

#print(image.format)
#print(image.size)
#print(image.mode)

#config1  = CycleGANConfig()
#config1.weights_gen_AB = "Gen_A_to_B.pth"
#config1.weights_gen_BA = "Gen_B_to_A.pth"
#config1.weights_dis_A = "Dis_A.pth"
#config1.weights_dis_B = "Dis_B.pth"
#config1.input_dims = (1,512,512)

#md = CycleGAN(config1)
#mri = md.predict(image)
#mri = mri.squeeze(0)
#print(mri.shape)

import torchvision.transforms as T

#transform = T.ToPILImage()
#img = transform(mri)
#print(type(img))

from torchvision.utils import save_image


#img1 = mri[0]
# img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
#save_image(img1, 'img1.png')
