# This code was copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and adapted for our purpose accordingly.

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
from models.networks import get_scheduler



if __name__ == '__main__':

    opt = Options().parseTrain()   # get training options

    # opt.phase = 'val'
    # val_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    opt.phase = 'train'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.schedulers = [get_scheduler(optimizer, opt) for optimizer in model.optimizers]

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
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if i % opt.update_TB_images_freq == 1: # Update inferences in tensorboard every 'opt.update_TB_images_freq' iterations
                with torch.no_grad():
                    opt.tbWriter.add_image('Train/RealA', torch.round((model.real_A[0] + 1.) / 2.0 * 255.0).byte(), i)
                    opt.tbWriter.add_image('Train/RealB', torch.round((model.real_B[0] + 1.) / 2.0 * 255.0).byte(), i)
                    opt.tbWriter.add_image('Train/Fake_A2B', torch.round((model.fake_A2B[0] + 1.) / 2.0 * 255.0).byte(), i)
                    opt.tbWriter.add_image('Train/Fake_B2A', torch.round((model.fake_B2A[0] + 1.) / 2.0 * 255.0).byte(), i)
                    opt.tbWriter.add_image('Train/Recon_A2B2A', torch.round((model.fake_A2B2A[0] + 1.) / 2.0 * 255.0).byte(), i)
                    opt.tbWriter.add_image('Train/Recon_B2A2B', torch.round((model.fake_B2A2B[0] + 1.) / 2.0 * 255.0).byte(), i)
                    if opt.use_segm_model:
                        labelMap = torch.argmax(model.rec_B_Segm[0], 0).byte() * (255 / 7)
                        opt.tbWriter.add_image('Train/Rec_B2A2B_Seg',torch.stack([labelMap, labelMap, labelMap], dim=0), i)
                        labelMap = torch.argmax(model.idt_B_Segm[0], 0).byte() * (255 / 7)
                        opt.tbWriter.add_image('Train/Idt_B2B_Seg',torch.stack([labelMap, labelMap, labelMap], dim=0), i)

            currLossesDict = {}
            currLossesDict['G_A'] = model.G_loss_A.item()
            currLossesDict['G_B'] = model.G_loss_B.item()
            currLossesDict['D_A'] = model.D_loss_A.item()
            currLossesDict['D_B'] = model.D_loss_B.item()
            opt.tbWriter.add_scalars('Plot/train', currLossesDict, i)
            opt.logger.info('[Iteration {}] '.format(i) + printDict(currLossesDict))

            if i % opt.saveModelEachNIteration == 0:
                opt.logger.info('Saving model at iteration {}'.format(str(i)))
                for name in model.model_names_save:
                    if isinstance(name, str):
                        save_filename = '%s_net_%s.pth' % (str(i), name)
                        save_path = os.path.join(model.save_dir, save_filename)
                        net = getattr(model, name)
                        if len(model.gpu_ids) > 0 and torch.cuda.is_available():
                            torch.save(net.cpu().state_dict(), save_path)
                            net.cuda(model.gpu_ids[0])
                        else:
                            torch.save(net.cpu().state_dict(), save_path)

            if i % opt.print_memory_usage_freq == 0:
                # opt.logger.info('[Iteration ' + str(i) + '] ' + parse_nvidia_smi(opt.gpu_ids[0]))
                opt.logger.info('[Iteration ' + str(i) + '] ' + parse_RAM_info())

            model.update_learning_rate(i)  # update learning rates at the end of every iteration.

        opt.logger.info('########################## TRAINING DONE ##########################')

    except:
        opt.logger.exception('! Exception !')
        raise

