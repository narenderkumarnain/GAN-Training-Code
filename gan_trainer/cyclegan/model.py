"""
For External Use.
Wrapper Model class for CycleGAN.
"""
import wandb
import logging
import torch
import torch.nn as nn
from gan_trainer.cyclegan.config import CycleGANConfig
from gan_trainer.cyclegan.networks import Generator, Discriminator

# some configurations for logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()


class CycleGAN:
    """
    Class to encapsulate CycleGAN Working.
    Config Class Parameters controls various Functionalities Here.
    """

    def __init__(self, config: CycleGANConfig):
        """

        Parameters
        ----------
        config: Configuration for Model.
        """
        self.config = config
        if not self.config.inference:
            # two generators
            self.Gen_A_to_B = Generator(self.config.input_dims[0],self.config.residual_blocks)
            self.Gen_B_to_A = Generator(self.config.input_dims[0], self.config.residual_blocks)
            # two discriminators
            self.Dis_A = Discriminator(self.config.input_dims[0])
            self.Dis_B = Discriminator(self.config.input_dims[0])

            # loading weights
            if self.config.weights_gen_AB is not None:
                self.Gen_A_to_B.load_state_dict(torch.load(self.config.weights_gen_AB))
                logger.info(f"Gen_A_to_B: weights loaded from {self.config.weights_gen_AB}")
            if self.config.weights_gen_BA is not None:
                self.Gen_B_to_A.load_state_dict(torch.load(self.config.weights_gen_BA))
                logger.info(f"Gen_B_to_A: weights loaded from {self.config.weights_gen_BA}")
            if self.config.weights_dis_A is not None:
                self.Dis_A.load_state_dict(torch.load(self.config.weights_dis_A))
                logger.info(f"Dis A: weights loaded from {self.config.weights_dis_A}")
            if self.config.weights_dis_B is not None:
                self.Dis_B.load_state_dict(torch.load(self.config.weights_dis_B))
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


    def predict(self,img):
        """

        Parameters
        ----------
        img: Input Image

        Returns
        -------

        """
        pass

    def train(self,dataset, epochs = None):
        device = torch.device(self.config.device) if self.config.device.contains("gpu") else torch.device("cpu")

        logger.info(f"Model in Training")
        # pulling models to device
        self.Gen_A_to_B = self.Gen_A_to_B.to(device)
        self.Gen_B_to_A = self.Gen_B_to_A.to(device)
        self.Dis_B = self.Dis_B.to(device)
        self.Dis_A = self.Dis_A.to(device)

        # losses for gan training
        lossGAN = nn.MSELoss().to(device)
        lossCycle = nn.L1Loss().to(device)
        lossIdentity = nn.L1Loss().to(device)

    def train_TPU(self):
        pass
    