"""
For External Use.
Wrapper Model class for CycleGAN.
"""
import os.path
import wandb
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
from gan_trainer.cyclegan.utils import ReplayBuffer, LambdaLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from gan_trainer.cyclegan.config import CycleGANConfig
from gan_trainer.cyclegan.networks import Generator, Discriminator
from gan_trainer.cyclegan.dataset import ImagetoImageDataset

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
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        if self.config.input_dims[0] != 1:
            self.transform += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        wandb.init(project = project_name)
        if not self.config.inference:
            # two generators
            self.Gen_A_to_B = Generator(self.config.input_dims[0],self.config.residual_blocks)
            self.Gen_B_to_A = Generator(self.config.input_dims[0], self.config.residual_blocks)
            # two discriminators
            self.Dis_A = Discriminator(self.config.input_dims[0])
            self.Dis_B = Discriminator(self.config.input_dims[0])

            wandb.watch((self.Gen_A_to_B, self.Gen_B_to_A, self.Dis_A, self.Dis_B), log = "all", log_freq = config.log_freq)

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


    def predict(self,img, A_to_B = True):
        """

        Parameters
        ----------
        img: Input Image

        Returns
        -------

        """
        transform = transforms.Compose(self.transform)
        img = transform(img)
        if A_to_B:
            res = self.Gen_A_to_B(img)
        else:
            res = self.Gen_B_to_A(img)

        return res

    def train(self):
        device = torch.device(self.config.device)
        logger.info(device)
        logger.info(f"Model in Training")

        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
            logger.info(f'Directory {self.config.checkpoint_dir} Created!')
        if not os.path.exists(self.config.sample_results_dir):
            os.mkdir(self.config.sample_results_dir)
            logger.info(f'Directory {self.config.sample_results_dir} Created!')
        # pulling models to device
        self.Gen_A_to_B = self.Gen_A_to_B.to(device)
        self.Gen_B_to_A = self.Gen_B_to_A.to(device)
        self.Dis_B = self.Dis_B.to(device)
        self.Dis_A = self.Dis_A.to(device)

        # losses for gan training
        lossGAN = nn.MSELoss().to(device)
        lossCycle = nn.L1Loss().to(device)
        lossIdentity = nn.L1Loss().to(device)

        # optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.Gen_A_to_B.parameters(), self.Gen_B_to_A.parameters()), lr=self.config.lr, betas=(self.config.b1, self.config.b2)
        )
        optimizer_D_A = torch.optim.Adam(self.Dis_A.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))
        optimizer_D_B = torch.optim.Adam(self.Dis_B.parameters(), lr=self.config.lr, betas=(self.config.b1, self.config.b2))

        # Learning rate update schedulers
        # lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer_G, lr_lambda=LambdaLR(self.config.n_epochs, 0, self.config.decay_epoch).step
        # )
        # lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer_D_A, lr_lambda=LambdaLR(self.config.n_epochs, 0, self.config.decay_epoch).step
        # )
        # lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer_D_B, lr_lambda=LambdaLR(self.config.n_epochs, 0, self.config.decay_epoch).step
        # )

        is_monochrome = self.config.input_dims[0] == 1
        # loading the dataloader
        trainLoader = DataLoader(
            ImagetoImageDataset(self.config.dataset_path,mode="train",sub_dir=self.config.sub_dir_dataset,imageTransforms=self.transform,monochrome=is_monochrome),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

        testLoader = DataLoader(
            ImagetoImageDataset(self.config.dataset_path,mode="test",sub_dir=self.config.sub_dir_dataset,imageTransforms=self.transform,monochrome=is_monochrome),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Buffers of previously generated samples
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        # training loop starts
        previousTime = time.time()

        for epoch in tqdm.tqdm(range(self.config.n_epochs),position = 0, leave = True,  desc=f"Epoch"):
            # for each batch
            for i, batch in enumerate(tqdm.tqdm(trainLoader,position = 0, leave = True,desc=f"Batch")):
                img_real_A,img_real_B = batch

                img_real_A = img_real_A.to(device)
                img_real_B = img_real_B.to(device)

                logger.info(f'Image size {img_real_A.shape} {img_real_B.shape}')

                # Adversarial ground truths
                channels, height, width = self.config.input_dims
                dis_output_shape = (1, height // 2 ** 4, width // 2 ** 4)
                # if self.config.input_dims[0] == 1:
                #     dis_output_shape = (1, height // 2 ** 4 - 1, width // 2 ** 4 - 1)
                valid = Variable(torch.Tensor(np.ones((img_real_A.size(0), *dis_output_shape))), requires_grad=False).to(device)
                fake = Variable(torch.Tensor(np.zeros((img_real_A.size(0), *dis_output_shape))), requires_grad=False).to(device)

                # Generator training
                self.Gen_B_to_A.train()
                self.Gen_A_to_B.train()

                optimizer_G.zero_grad()

                # GAN Losses
                img_fake_B = self.Gen_A_to_B(img_real_A)

                self.Dis_B.eval()
                self.Dis_A.eval()

                # logger.info(f'Size 1 :{self.Dis_B(img_fake_B).shape} Size 2 {valid.shape } ')
                loss_GAN_AB = lossGAN(self.Dis_B(img_fake_B),valid)

                img_fake_A = self.Gen_B_to_A(img_fake_B)
                loss_GAN_BA = lossGAN(self.Dis_A(img_fake_A),valid)

                loss_GAN_net = (loss_GAN_BA+loss_GAN_AB)/2

                self.Dis_B.train()
                self.Dis_A.train()

                # Cycle Losses
                recovered_A = self.Gen_B_to_A(img_fake_B)
                recovered_B = self.Gen_A_to_B(img_fake_A)

                # logger.info(f'loss cycle a {recovered_A.shape}  {img_real_A.shape }  {img_fake_B.shape}')
                loss_cycle_A = lossCycle(recovered_A, img_real_A)
                loss_cycle_B = lossCycle(recovered_B, img_real_B)

                loss_cycle = (loss_cycle_A+loss_cycle_B) / 2

                # Identity Loss for Tint restoration
                loss_identity_A = lossIdentity(self.Gen_B_to_A(img_real_A), img_real_A)
                loss_identity_B = lossIdentity(self.Gen_A_to_B(img_real_B), img_real_B)

                loss_identity = (loss_identity_B+loss_identity_A) / 2

                # Total Loss
                loss_Gen = loss_GAN_net + self.config.lambda_cyclic * loss_cycle + self.config.lambda_identity * loss_identity

                loss_Gen.backward()

                optimizer_G.step()


                # Discriminator Training

                # Discriminator A
                optimizer_D_A.zero_grad()

                loss_real = lossGAN(self.Dis_A(img_real_A), valid)
                fake_A_prev = fake_A_buffer.push_and_pop(img_fake_A)
                loss_fake = lossGAN(self.Dis_A(fake_A_prev.detach()), fake)

                loss_DA = (loss_fake  + loss_real) / 2
                loss_DA.backward()
                optimizer_D_A.step()

                # Discriminator B
                optimizer_D_B.zero_grad()

                loss_real = lossGAN(self.Dis_B(img_real_B), valid)
                fake_B_prev = fake_A_buffer.push_and_pop(img_fake_B)
                loss_fake = lossGAN(self.Dis_B(fake_B_prev.detach()), fake)

                loss_DB = (loss_real+loss_fake) / 2

                loss_DB.backward()
                optimizer_D_B.step()

                # Determine approximate time left
                batches_done = epoch * len(trainLoader) + i
                batches_left = self.config.n_epochs * len(trainLoader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - previousTime))
                previousTime = time.time()

                # Logging The losses in wandb and logger

                # if batches_done % self.config.log_freq == 0:
                #     logger.info(f'''
                #     Epoch:{epoch} Batch:{i}
                #     Loss Generator AB: {round(loss_Gen.item(),4)}
                #         Cyclic:{round(loss_cycle.item(),4)} Identity:{round(loss_identity.item(),4)} GAN:{round(loss_GAN_net.item(),4)}
                #     Loss Discriminator DA:{round(loss_DA.item(),4)}
                #     Loss Discriminator DB:{round(loss_DB.item(),4)}
                #     Remaining Time: {time_left}
                #     ''')

                wandb.log({
                    'Loss Generator': round(loss_Gen.item(),4) ,
                    'Loss Generator Cyclic': round(loss_cycle.item(),4) ,
                    'Loss Generator GAN': round(loss_GAN_net.item(),4),
                    'Loss Generator Identity': round(loss_identity.item(),4),
                    'Loss Dis A': round(loss_DA.item(),4) ,
                    'Loss Dis B': round(loss_DB.item(),4)
                })

                # saving some sample results
                if batches_done % self.config.batch_freq == 0:
                    imgs = next(iter(testLoader))
                    self.Gen_A_to_B.eval()
                    self.Gen_B_to_A.eval()
                    real_A = imgs[0].to(device)
                    fake_B = self.Gen_A_to_B(real_A)
                    real_B = imgs[1].to(device)
                    fake_A = self.Gen_B_to_A(real_B)
                    # Arange images along x-axis
                    real_A = make_grid(real_A, nrow=5, normalize=True)
                    real_B = make_grid(real_B, nrow=5, normalize=True)
                    fake_A = make_grid(fake_A, nrow=5, normalize=True)
                    fake_B = make_grid(fake_B, nrow=5, normalize=True)
                    # Arange images along y-axis
                    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
                    save_image(image_grid, os.path.join(self.config.sample_results_dir, f"epoch{epoch}batch{i}.png"), normalize=False)
                    logger.info(f'Sample Results Saved: {os.path.join(self.config.sample_results_dir, f"epoch{epoch}batch{i}.png")}')

                # Checkpoint
                if batches_done % self.config.checkpoint_freq == 0:
                    logger.info(f'Saving Checkpoints')
                    torch.save(self.Gen_A_to_B.state_dict(),
                                os.path.join(self.config.checkpoint_dir, "Gen_A_to_B.pth"))
                    torch.save(self.Gen_B_to_A.state_dict(),
                                os.path.join(self.config.checkpoint_dir, "Gen_B_to_A.pth"))
                    torch.save(self.Dis_A.state_dict(), os.path.join(self.config.checkpoint_dir, "Dis_A.pth"))
                    torch.save(self.Dis_B.state_dict(), os.path.join(self.config.checkpoint_dir, "Dis_B.pth"))
                # Update learning rates
                # lr_scheduler_G.step()
                # lr_scheduler_D_A.step()
                # lr_scheduler_D_B.step()







    def train_TPU(self):
        pass
    