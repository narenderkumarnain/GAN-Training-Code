"""
For External Use.
Wrapper Model class for CycleGAN.
"""
import torch
from gan_trainer.cyclegan.config import CycleGANConfig
from gan_trainer.cyclegan.networks import Generator, Discriminator

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
            if self.config.weights_gen_BA is not None:
                self.Gen_B_to_A.load_state_dict(torch.load(self.config.weights_gen_BA))
            if self.config.weights_dis_A is not None:
                self.Dis_A.load_state_dict(torch.load(self.config.weights_dis_A))
            if self.config.weights_dis_B is not None:
                self.Dis_B.load_state_dict(torch.load(self.config.weights_dis_B))

        else:
            if self.config.input_A:
                self.Gen_A_to_B = Generator(self.config.input_dims[0], self.config.residual_blocks)
                if self.config.weights_gen_AB is not None:
                    self.Gen_A_to_B.load_state_dict(torch.load(self.config.weights_gen_AB))
            else:
                self.Gen_B_to_A = Generator(self.config.input_dims[0], self.config.residual_blocks)
                if self.config.weights_gen_BA is not None:
                    self.Gen_B_to_A.load_state_dict(torch.load(self.config.weights_gen_BA))


    def predict(self,img):
        """

        Parameters
        ----------
        img: Input Image

        Returns
        -------

        """
        pass

    def train(self,epochs = None):
        pass

    def train_TPU(self):
        pass
    