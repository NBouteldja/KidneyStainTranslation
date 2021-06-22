# This code was copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and adapted for our purpose accordingly.
import torch
import numpy as np
import torchvision.transforms.functional as F
import random

class RangeNormalization(object):
    def __call__(self, sample):
        img, lbl = sample
        return img / 255.0 * 2.0 - 1.0, lbl


class ToTensor(object):
    def __call__(self, sample):
        img, lbl = sample

        #lbl = torch.from_numpy(lbl).long()
        img = torch.from_numpy(np.array(img, np.float32).transpose(2, 0, 1))

        return img, lbl


class RandomRotation(object):
    def __init__(self,angles):
        self. angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)

class RandomGamma(object):
    def __init__(self,delta):
        self.delta = delta

    def __call__(self, img):
        p = random.gauss(1,self.delta)
        if p <= 0.5: p = 0.5
        elif p>= 1.5: p = 1.5
        return F.adjust_gamma(img, p)
