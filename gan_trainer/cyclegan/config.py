"""
For External Use.
Configuration Arguments for CycleGAN Model Training.
"""

class CycleGANConfig():
    """
    Variables and default Values are defined here.
    """

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

