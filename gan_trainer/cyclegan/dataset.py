"""
For External Use.
Dataset class for our Datasets.
"""
import os
import glob
import torch
import random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# class for Image to Image Translation Datasets
class ImagetoImageDataset(Dataset):

    def __init__(self, root_dir, mode="train", imageTransforms=None, aligned=True, A_name="A", B_name="B", sub_dir = True, monochrome = False ):
        """
        Required Directory Structure:
        root_dir/
                train/
                    A/
                    B/
                test/
                    A/
                    B/

        Parameters
        ----------
        root_dir
        mode
        imageTransforms
        unaligned
        """
        self.aligned = aligned
        self.monochrome = monochrome
        self.transform = transforms.Compose(imageTransforms)

        if not sub_dir:
            self.images_A = sorted(glob.glob(os.path.join(root_dir, mode + A_name) + "/*.*"))
            self.images_B = sorted(glob.glob(os.path.join(root_dir, mode + B_name) + "/*.*"))
        else:
            self.images_A = sorted(glob.glob(os.path.join(root_dir, mode, A_name) + "/*.*"))
            self.images_B = sorted(glob.glob(os.path.join(root_dir, mode, B_name) + "/*.*"))


    def __len__(self):
        if self.aligned:
            return min(len(self.images_A), len(self.images_B))
        else:
            return max(len(self.images_A), len(self.images_B))


    def __getitem__(self, item):
        img_A = Image.open(self.images_A[item % len(self.images_A)])
        if self.aligned:
            img_B = Image.open(self.images_B[item % len(self.images_B)])
        else:
            img_B = Image.open(self.images_B[random.randint(0,len(self.images_B)) - 1])

        if not self.monochrome:
            img_A = img_A.convert('RGB')
            img_B = img_B.convert('RGB')
        else:
            img_A = ImageOps.grayscale(img_A)
            img_B = ImageOps.grayscale(img_B)

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        # return (np.array(img_A),np.array(img_B))
        return (img_A,img_B)






# if __name__ == "__main__":
#     transform = [
#         transforms.Resize((256, 256))
#     ]
#
#     dataset = ImagetoImageDataset('../../dataset/archive/apple2orange/apple2orange', 'train', sub_dir=False,imageTransforms=transform)
#     print(len(dataset))
#     print(dataset[100][0])
