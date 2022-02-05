"""
For External Use.
Configuration Arguments for CycleGAN Model Training.
"""
import logging

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

    def __str__(self):
        """
        String representation for Mode Config
        Returns
        -------

        """
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


if __name__=="__main__":
    config = CycleGANConfig()
    logger.info(config)



