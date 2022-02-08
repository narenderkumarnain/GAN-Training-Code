"""
For External Use.
Configuration Arguments for CycleGAN Model Training.
"""
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


if __name__=="__main__":
    config = CycleGANConfig()
    logger.info(config)



