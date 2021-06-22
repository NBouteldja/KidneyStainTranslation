# This code was copied from https://github.com/znxlwm/UGATIT-pytorch and adapted for our purpose accordingly.
# From: https://github.com/znxlwm/UGATIT-pytorch
# Author: Junho Kim
# Paper: U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation
# License: MIT License

import torch
import itertools
from image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .radam import RAdam
from .model import Custom
from .nutils import WeightedL1Loss


class UGATIT2Model(BaseModel):
    """
    This class implements the UGATIT model, for learning image-to-image translation without paired data.
    U-GAT-IT paper: https://arxiv.org/abs/1907.10830
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default U_GAT_IT did not use dropout
        parser.set_defaults(norm='instance')  # default U_GAT_IT did use instance norm
        if is_train:
            parser.add_argument('--weight_decay', type=float, default=0.0001)

            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_Seg', type=float, default=.5, help='weight for cycle loss (Seg(A), A -> B -> A, Seg(A))')
            parser.add_argument('--lambda_id', type=float, default=1.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_cam', type=int, default=1000, help='Weight for CAM')

            parser.add_argument('--n_layers_D_G', type=int, default=7, help='depth of global discriminators')
            parser.add_argument('--n_layers_D_L', type=int, default=5, help='depth of local discriminators')

            parser.add_argument('--lightFC', action='store_true', help='use light FC version which is input resolution independent')
        return parser

    def __init__(self, opt):
        """Initialize the U_GAT_IT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.model_names = ['genA2B', 'genB2A', 'disGA', 'disGB', 'disLA', 'disLB']
        self.model_names_save = ['genA2B', 'genB2A']

        if opt.netG.startswith('unet'):
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
            self.genA2B = UnetGenerator(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, n_downsampling=int(opt.netG.split('_')[-1]), norm_layer=norm_layer, use_dropout=False).to(self.device)
            self.genB2A = UnetGenerator(input_nc=opt.output_nc, output_nc=opt.input_nc, ngf=opt.ngf, n_downsampling=int(opt.netG.split('_')[-1]), norm_layer=norm_layer, use_dropout=False).to(self.device)
        else: #ResnetGenerator
            self.genA2B = ResnetGenerator(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, n_blocks=int(opt.netG.split('_')[-1][:-6]),n_downsampling=int(opt.netG.split('_')[1]), img_size=opt.crop_size, lightFC=opt.lightFC).to(self.device)
            self.genB2A = ResnetGenerator(input_nc=opt.output_nc, output_nc=opt.input_nc, ngf=opt.ngf, n_blocks=int(opt.netG.split('_')[-1][:-6]),n_downsampling=int(opt.netG.split('_')[1]), img_size=opt.crop_size, lightFC=opt.lightFC).to(self.device)

        if self.isTrain:
            self.disGA = Discriminator(input_nc=3, ndf=opt.ndf, n_layers=7).to(self.device)
            self.disGB = Discriminator(input_nc=3, ndf=opt.ndf, n_layers=7).to(self.device)
            self.disLA = Discriminator(input_nc=3, ndf=opt.ndf, n_layers=5).to(self.device)
            self.disLB = Discriminator(input_nc=3, ndf=opt.ndf, n_layers=5).to(self.device)

            """ Define Loss """
            self.L1_loss = torch.nn.L1Loss().to(self.device)
            self.MSE_loss = torch.nn.MSELoss().to(self.device)
            self.BCE_loss = torch.nn.BCEWithLogitsLoss().to(self.device)

            """ Trainer """
            self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

            self.optimizers.append(self.D_optim)
            self.optimizers.append(self.G_optim)

            """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
            self.Rho_clipper = RhoClipper(0, 1)

            if self.opt.use_segm_model:
                self.segm_model = Custom(input_ch=3, output_ch=8, modelDim=2)
                self.segm_model.load_state_dict(torch.load(opt.segm_model_path))
                opt.logger.info('### Segmentation Model Loaded ###')

                self.segm_model.to(self.device)
                self.set_requires_grad(self.segm_model, False)
                self.segm_model.train(False)

                self.segmentationloss = WeightedL1Loss(weights = torch.FloatTensor([1., 1., 1., 1., 1., 10., 1., 10.]).to(self.device))


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.use_segm_model:
            if self.opt.dataset_mode == 'seg_map':
                self.segm_B = input['segm_B'].to(self.device)
            else:
                with torch.no_grad():
                    self.segm_B = self.segm_model(self.real_B)


    def optimize_parameters(self):
        identity_weight = self.opt.lambda_id
        cycle_weight_A = self.opt.lambda_A
        cycle_weight_B = self.opt.lambda_B
        cam_weight = self.opt.lambda_cam
        seg_weight = self.opt.lambda_Seg

        self.D_optim.zero_grad()

        fake_A2B, _, _ = self.genA2B(self.real_A)
        fake_B2A, _, _ = self.genB2A(self.real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(self.real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(self.real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(self.real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
        D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit,torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
        D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit,torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
        D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit,torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
        D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit,torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

        self.D_loss_A = D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA
        self.D_loss_B = D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB

        Discriminator_loss = self.D_loss_A + self.D_loss_B
        Discriminator_loss.backward()
        self.D_optim.step()

        # Update G
        self.G_optim.zero_grad()

        self.fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap = self.genA2B(self.real_A)
        self.fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap = self.genB2A(self.real_B)

        self.fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(self.fake_A2B)
        self.fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(self.fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(self.real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(self.fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(self.fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(self.fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(self.fake_A2B)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

        G_recon_loss_A = self.L1_loss(self.fake_A2B2A, self.real_A)
        G_recon_loss_B = self.L1_loss(self.fake_B2A2B, self.real_B)

        G_identity_loss_A = self.L1_loss(fake_A2A, self.real_A)
        G_identity_loss_B = self.L1_loss(fake_B2B, self.real_B)

        G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit,torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
        G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit,torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

        self.G_loss_A = G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA + cycle_weight_A * G_recon_loss_A + identity_weight * G_identity_loss_A + cam_weight * G_cam_loss_A
        self.G_loss_B = G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB + cycle_weight_B * G_recon_loss_B + identity_weight * G_identity_loss_B + cam_weight * G_cam_loss_B

        if self.opt.use_segm_model:
            self.rec_B_Segm = self.segm_model(self.fake_B2A2B) #Segm(G_B(G_A(A)))
            self.idt_B_Segm = self.segm_model(fake_B2B)
            self.G_loss_B += seg_weight * (self.segmentationloss(self.rec_B_Segm, self.segm_B) + self.segmentationloss(self.idt_B_Segm, self.segm_B))

        Generator_loss = self.G_loss_A + self.G_loss_B  # + G_heatmap_loss * 10

        Generator_loss.backward()
        self.G_optim.step()

        # clip parameter of AdaILN and ILN, applied after optimizer step
        self.genA2B.apply(self.Rho_clipper)
        self.genB2A.apply(self.Rho_clipper)

    def forward(self):
        return 0

    def computeLosses(self):
        return 0

    def perform_test_conversion(self, input):
        if self.opt.direction == 'AtoB':
            res, _, _ = self.genA2B(input)
            return res
        else:
            res, _, _ = self.genB2A(input)
            return res



import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import functools

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, n_downsampling = 2, img_size=640, lightFC=True):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.lightFC = lightFC

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.lightFC:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False), nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False), nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False), nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False), nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        if self.lightFC:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=6, norm_layer=None, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        self.unet_block_First = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=self.unet_block_First, norm_layer=norm_layer, use_dropout=use_dropout)
        for i in range(n_downsampling - 6):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        res = self.model(input)
        return res, self.unet_block_First.cam_logit, self.unet_block_First.heatmap


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            #conv = nn.Conv2d(inner_nc,outer_nc,kernel_size=5,stride=1,padding=2 )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up #+ [conv]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            # Class Activation Map
            self.gap_fc = nn.Linear(inner_nc, 1, bias=False)
            self.gmp_fc = nn.Linear(inner_nc, 1, bias=False)
            self.conv1x1 = nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=1, stride=1, bias=True)
            self.relu = nn.ReLU(False)

            self.up = nn.Sequential(*[uprelu, upconv, upnorm])
            model = down
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.innermost:
            bottomX = self.model(x)
            gap = torch.nn.functional.adaptive_avg_pool2d(bottomX, 1)
            gap_logit = self.gap_fc(gap.view(bottomX.shape[0], -1))
            gap_weight = list(self.gap_fc.parameters())[0]
            gap = bottomX * gap_weight.unsqueeze(2).unsqueeze(3)

            gmp = torch.nn.functional.adaptive_max_pool2d(bottomX, 1)
            gmp_logit = self.gmp_fc(gmp.view(bottomX.shape[0], -1))
            gmp_weight = list(self.gmp_fc.parameters())[0]
            gmp = bottomX * gmp_weight.unsqueeze(2).unsqueeze(3)

            self.cam_logit = torch.cat([gap_logit, gmp_logit], 1)
            bottomX = torch.cat([gap, gmp], 1)
            bottomX = self.relu(self.conv1x1(bottomX))

            self.heatmap = torch.sum(bottomX, dim=1, keepdim=True)

            return torch.cat([x, self.up(bottomX)], 1)
        elif self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)