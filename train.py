"""As a whole, this code framework is inspired and heavily based on the great CylceGAN/pix2pix Repository from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. We copied and adapted several code fragments.
The specified repo was written by Jun-Yan Zhu and Taesung Park, and supported by Tongzhou Wang.
"""
# From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Author: Liyuan Liu
# Paper: CycleGAN (https://arxiv.org/pdf/1703.10593.pdf) and pix2pix (https://arxiv.org/pdf/1611.07004.pdf)
# License: BSD License


import numpy as np
import os
import math
import time
import sys
import logging as log

import torch

from options import Options
from data import create_dataset
from models import create_model

from utils import parse_RAM_info, parse_nvidia_smi, visualizeForwardsNoGrad, printDict




if __name__ == '__main__':

    opt = Options().parseTrain()   # get training options

    # opt.phase = 'val'
    # val_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    opt.phase = 'train'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    loader_iter = iter(dataset)

    opt.logger.info('################################ TRAINING STARTS ################################ - 1 Epoch = {} iterations'.format(math.ceil(dataset.__len__() / opt.batch_size)))

    try:
        for i in range(1, opt.niters+1):
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataset)
                data = next(loader_iter)

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            if i < opt.niters_init:
                model.optimize_gen_for_id_loss() # update generators using only identity losses in first iterations
            else:
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if i % opt.update_TB_images_freq == 1: # Update inferences in tensorboard every 'opt.update_TB_images_freq' iterations
                visualizeForwardsNoGrad(model, iteration=i, tbWriter=opt.tbWriter, phase='Train')

            currLossesDict = model.get_current_losses()
            opt.tbWriter.add_scalars('Plot/train', currLossesDict, i)
            opt.logger.info('[Iteration {}] '.format(i) + printDict(currLossesDict))

            if i % opt.saveModelEachNIteration == 0:
                opt.logger.info('Saving model at iteration {}'.format(i))
                model.save_networks(str(i))

            if i % opt.print_memory_usage_freq == 0:
                # opt.logger.info('[Iteration ' + str(i) + '] ' + parse_nvidia_smi(opt.gpu_ids[0]))
                opt.logger.info('[Iteration ' + str(i) + '] ' + parse_RAM_info())

            model.update_learning_rate(i)  # update learning rates at the end of every iteration.

        opt.logger.info('########################## TRAINING DONE ##########################')

    except:
        opt.logger.exception('! Exception !')
        raise

