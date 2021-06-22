# This code was copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and adapted for our purpose accordingly.
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/stain/trainA'
    and  from domain B '/path/to/data/PAS/trainB' (B is PAS) respectively
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.stain,  opt.phase)  # create a path '/path/to/data/stainA/train'
        self.A_paths, self.A_imgs = make_dataset(self.dir_A, opt.preload, opt.load_size, opt.max_dataset_size)   # load images from '/path/to/data/trainA'

        self.dir_B = os.path.join(opt.dataroot, opt.stainB, opt.phase)  # create a path '/path/to/data/stainB/train'
        self.B_paths, self.B_imgs = make_dataset(self.dir_B, opt.preload, opt.load_size, opt.max_dataset_size)    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        opt.logger.info('Size of '+ opt.phase +' data set '+ opt.stain +' (A): ' + str(self.A_size))
        opt.logger.info('Size of '+ opt.phase +' data set '+ opt.stainB +' (B): ' + str(self.B_size))

        self.transform_A = get_transform(self.opt, grayscale=(opt.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(opt.output_nc == 1))
        self.preload = opt.preload
        self.resize = (opt.load_size,opt.load_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- (its corresponding) image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        if self.preload:
            A_img = self.A_imgs[index % self.A_size]
            B_img = self.B_imgs[index_B]
        else:
            A_img = Image.open(A_path).convert('RGB').resize(self.resize, Image.BILINEAR)
            B_img = Image.open(B_path).convert('RGB').resize(self.resize, Image.BILINEAR)
            # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
