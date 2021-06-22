import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial
import math





####################################### XAVIER WEIGHT INIT #########################################
def init_weights_xavier_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)




####################################################################################################
# ----- Modified Unet from segmentation paper ----- #
class Custom(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(Custom, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return torch.softmax(x, 1)


class initialconv(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(initialconv, self).__init__()
        self.conv = conv_block(in_ch, out_ch, modelDim)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(down, self).__init__()
        if modelDim == 2:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                # nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, groups=in_ch),
                conv_block(in_ch, out_ch, modelDim)
            )
        elif modelDim == 3:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                conv_block(in_ch, out_ch, modelDim)
            )
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x



############################### don't forget about spatial size changes: ################################
# max pool (kernel=2, stride=2) -> Input: (10x5) -> Output: (5x2)
# torch.nn.Conv2d(1,1,3,stride=2,padding=1) -> Input: (10x5) -> Output: (5x3)
# nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) -> Input: (10x5) -> Output: (20x10)
# nn.ConvTranspose2d(1, 1, 3, padding=1, stride=2) -> Input: (10x5) -> Output: (19x9) : *2-1
# nn.ConvTranspose2d(1, 1, 3, padding=1, stride=2, output_padding=1) -> Input: (10x5) -> Output: (20x10) : *2
# nn.ConvTranspose2d(1, 1, 2, padding=0, stride=2) -> Input: (10x5) -> Output: (20x10) : *2
# nn.ConvTranspose2d(1, 1, 2, padding=0, stride=2, output_padding=1) -> Input: (10x5) -> Output: (21x11) : *2+1
# => Vanilla Unet & nnUnet => Both use max pooling (in encoder) and transposed conv (in decoder)!
#########################################################################################################
class up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False):
        super(up, self).__init__()
        self.modelDim = modelDim
        if modelDim == 2:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                # self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1//2, 2, padding=0, stride=2)
                self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        elif modelDim == 3:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                # self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1//2, 2, padding=0, stride=2)
                self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

        # self.conv = conv_block_noPadding(in_ch_1//2 + in_ch_2, out_ch, modelDim)
        self.conv = conv_block_noPadding(in_ch_1 + in_ch_2, out_ch, modelDim)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        x = torch.cat([x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3], x1], dim=1)
        x = self.conv(x)
        return x


class conv_block_noPadding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block_noPadding, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(outconv, self).__init__()
        if modelDim == 2:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        elif modelDim == 3:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.conv(x)
        return x



