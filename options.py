# This code was copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and adapted for our purpose accordingly.

import sys
import argparse
import os
import torch
import models
import data
import shutil
import logging as log

from tensorboardX import SummaryWriter


class Options:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to folder with the stain specific datafolders')
        parser.add_argument('--stain', required=True, help='stain-imagefolder in dataroot (should have subfolders trainA, valA) possible options:[ErHr3 | F4-80 | NGAL | Fibronectin | CD31 | CD44 |  CD45 | Ly6G | Col_I | ColIII | Col_IV | PAS]')
        parser.add_argument('--stainB', required=True, help='stain-imagefolder in dataroot (should have subfolders trainB, valB) possible options:[ErHr3 | F4-80 | NGAL | Fibronectin | CD31 | CD44 |  CD45 | Ly6G | Col_I | ColIII | Col_IV | PAS]')
        parser.add_argument('--resultsPath', required=True, help='path to base results folder')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix for model subfolders, e.g. _dropout => cycle_gan_..._dropout')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | U_GAT_IT | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_3_9blocks', help='specify generator architecture [resnet_X_Yblocks -> X=depth, Y= #blocks | unet_X -> X=depth')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--use_segm_model', action='store_true',help='use pretrained segmentation model')
        parser.add_argument('--segm_model_path', type=str, default ='<path-to-model.pt', help='path to the segmentation model for domain B' )
        parser.add_argument('--use_MC', action='store_true',help='use multi channel inputs')
        # training parameters
        parser.add_argument('--niters', type=int, default=20000, help='# of iterations at starting learning rate')
        parser.add_argument('--niters_init', type=int, default=0, help='number of iterations training with only identity loss')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--niters_linDecay', type=int, default=10000, help='[policy: linear] # of iterations to linearly decay learning rate to zero')
        parser.add_argument('--niters_stepDecay', type=int, default=50, help='[policy: step] multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--validation_freq', type=int, default=500, help='iteration frequency for validation')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--preload', action='store_true', help='if specified, the images will be put into RAM before training starts')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=640, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=640, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a randomly varying subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | resize | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        # network saving and loading parameters
        parser.add_argument('--saveModelEachNIteration', type=int, default=float("inf"), help='iteration frequency to save models')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # additional debugging parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print network architectures')
        parser.add_argument('--print_memory_usage_freq', type=int, default=5000, help='iteration frequency to print RAM & VRAM usage info')
        parser.add_argument('--update_TB_images_freq', type=int, default=1000, help='iteration frequency to update inference images in tensorboard')


        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, opt.phase == 'train')
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, opt.phase == 'train')

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt, logger):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = 'Script call arguments are:\n\n' + ' '.join(sys.argv[1:]) + '\n\n'
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logger.info(message)


    def parseTrain(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        opt.model_foldername = self.setOptModelFolderName(opt)

        opt.finalResultsFolder = os.path.join(opt.resultsPath, opt.model_foldername, opt.stain + '_' + opt.stainB)
        opt.checkpoints_dir = os.path.join(opt.finalResultsFolder, 'Models')

        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)

        # Set up logger
        log.basicConfig(
            level=log.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y/%m/%d %I:%M:%S %p',
            handlers=[
                log.FileHandler(opt.finalResultsFolder + '/Train_LOGS.log', 'w'),
                log.StreamHandler(sys.stdout)
            ])
        logger = log.getLogger()

        self.print_options(opt, logger)
        # setting up tensorboard visualization
        tensorboardPath = os.path.join(opt.finalResultsFolder, 'TB')
        shutil.rmtree(tensorboardPath, ignore_errors=True)  # <- remove existing TB events
        opt.tbWriter = SummaryWriter(log_dir=tensorboardPath)
        opt.logger = logger

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        torch.backends.cudnn.benchmark = True

        return opt


    def parseTestStain(self, stain):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        opt.stain = stain

        opt.model_foldername = self.setOptModelFolderName(opt)

        opt.finalResultsFolder = os.path.join(opt.resultsPath, opt.model_foldername, opt.stain + '_' + opt.stainB)
        opt.checkpoints_dir = os.path.join(opt.finalResultsFolder, 'Models')
        opt.testResultsPath = os.path.join(opt.resultsPath, opt.model_foldername, 'TestResults')
        opt.testResultImagesPath = os.path.join(opt.testResultsPath, stain)

        if not os.path.exists(opt.testResultImagesPath):
            os.makedirs(opt.testResultImagesPath)

        return opt

    @staticmethod
    def setOptModelFolderName(opt):
        """Set opt.model_foldername specifying the folder of trained model"""

        model_foldername = opt.model + '_gen_' + opt.netG + '_' + str(opt.ngf) + '_dis_' + opt.netD + '_' + str(opt.n_layers_D) + '_' + str(
            opt.ndf) + '_batch_' + str(opt.batch_size) + '_init_' + opt.init_type + '_lr_' + str(
            opt.lr) + '_policy_' + opt.lr_policy + '_initItersId_' + str(opt.niters_init) + '_loss_' + opt.gan_mode + '_cropSize_' + str(opt.crop_size)

        if opt.model == 'cycle_gan':
            model_foldername += '_lamA_' + str(opt.lambda_A) + '_lamB_' + str(opt.lambda_B) + '_lamId_' + str(opt.lambda_id)

        elif opt.model == 'U_GAT_IT' or opt.model == 'U_GAT_IT_2':
            model_foldername += '_lamA_' + str(opt.lambda_A) + '_lamB_' + str(opt.lambda_B) + '_lamId_' + str(opt.lambda_id) + '_lamCam_' + str(
                opt.lambda_cam) + '_DlayersG' + str(opt.n_layers_D_G) + '_DlayersL' + str(opt.n_layers_D_L)

        if opt.use_segm_model:
            model_foldername += '_SegNet_lamSeg_' + str(opt.lambda_Seg)

        if opt.use_MC:
            model_foldername += '_useMC'

        if opt.suffix != '':
            model_foldername += opt.suffix

        return model_foldername


