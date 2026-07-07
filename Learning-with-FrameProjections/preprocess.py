#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import warnings

# Suppress verbose processing logs
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


# In[4]:


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing pipeline on device: {DEVICE}")

SAVE_DIR = "./data"
OUTPUT_FILE = os.path.join(SAVE_DIR, "sleep_combined.pt")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STAGE_MAPPING = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4
}
EPOCH_SEC = 30       
TARGET_SFREQ = 100   # 100Hz -> 3000 samples per epoch
TIME_STEPS = 3000    
LATENT_DIM = 128     
NUM_CLASSES = 5      
PRETRAIN_EPOCHS = 40
EVAL_EPOCHS = 30
BATCH_SIZE = 64


# In[ ]:


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


# In[ ]:


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


# In[ ]:


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)

        # shortcut
        out += identity

        return out



# In[5]:


class FourierEncoder(nn.Module):
    def __init__(self, in_channels, in_length, out_channels, kernel_size=3, initial_stride=1):
        super(FourierEncoder, self).__init__()

        # --- Amplitude branch ---
        # Process absolute values (amplitude)
        self.abs_conv = nn.Conv1d(in_channels, 16, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)
        self.abs_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.abs_block1 = BasicBlock(
            in_channels=16, out_channels=32, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=True
        )
        self.abs_block2 = BasicBlock(
            in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=False
        )

        # --- Phase branch ---
        # Process phase values (angle)
        self.ang_conv = nn.Conv1d(in_channels, 16, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)
        # self.ang_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.ang_conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)
        # self.ang_conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)

        self.ang_block1 = BasicBlock(
            in_channels=16, out_channels=32, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=True
        )
        self.ang_block2 = BasicBlock(
            in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=False
        )

        if in_length == 200:
            self.fc_abs = nn.Linear(26, 1)
            self.fc_angle = nn.Linear(26, 1)
        elif in_length == 128:
            self.fc_abs = nn.Linear(17, 1)
            self.fc_angle = nn.Linear(17, 1)
        elif in_length == 100:
            self.fc_abs = nn.Linear(13, 1)
            self.fc_angle = nn.Linear(13, 1)
        elif in_length == 480:
            self.fc_abs = nn.Linear(61, 1)
            self.fc_angle = nn.Linear(61, 1)
        elif in_length == 1000:
            self.fc_abs = nn.Linear(126, 1)
            self.fc_angle = nn.Linear(126, 1)
        elif in_length == 3004:
            self.fc_abs = nn.Linear(376, 1)
            self.fc_angle = nn.Linear(376, 1)
        elif in_length == 3000:
            self.fc_abs = nn.Linear(376, 1)
            self.fc_angle = nn.Linear(376, 1)
        # Fully connected layer combining both branches.
        # Each branch outputs 64 channels after global pooling.
        self.fc = nn.Linear(64, 128) # Isoalign
        # self.fc = nn.Linear(64, 192) # CLIP

    def forward(self, x):
        # Compute amplitude and phase from complex input.
        x_abs = torch.abs(x).float()    # shape: (B, C, T)
        x_ang = torch.angle(x).float()  # shape: (B, C, T)
        # Process amplitude branch.
        a = self.abs_conv(x_abs)
        a = F.relu(a)
        a = self.abs_block1(a)
        a = self.abs_block2(a)
        # Global average pooling over time dimension.
        a = self.fc_abs(a).squeeze(-1) # shape: (B, 64)

        # # Process phase branch.
        p = self.ang_conv(x_ang)
        p = F.relu(p)
        p = self.ang_block1(p)
        p = self.ang_block2(p)
        p = self.fc_angle(p).squeeze(-1) # shape: (B, 64)

        # Concatenate features from both branches.
        out = torch.cat([a, p], dim=1)  # shape: (B, 128)
        # out = self.fc(a)
        return out


# In[6]:


class ResNet1D(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
    """
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False, backbone=False, output_dim=200):
        super(ResNet1D, self).__init__()

        self.out_dim = output_dim
        self.backbone = backbone
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        self.dense2 = nn.Linear(out_channels, self.out_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(-1,-2) # RESNET 1D takes channels first
        out = x

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.backbone:
            out = self.dense2(out)
            return None, out
        # out = self.do(out)
        out_class = self.dense(out)
        # out = self.softmax(out)

        return out_class, out    


# In[ ]:


def conbr_block_2d(in_channels, out_channels, kernel_size, stride, padding):
    """A 2D convolutional block: Conv2d -> BatchNorm2d -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# In[7]:


class UNET_2D_simp(nn.Module):
    def __init__(self, input_channels, output_channels, layer_n, spect_freq, spect_time, kernel_size):
        """
        A simplified U-Net for 2D data (e.g. wavelet spectrograms).

        Parameters:
        - input_channels: Number of input channels (e.g., 1 for a single spectrogram channel)
        - output_channels: Number of output channels (e.g., number of classes or reconstruction channels)
        - layer_n: Base number of filters
        - kernel_size: Size of the convolution kernel (assumed square)
        - depth: (Not used in this simplified version, but could control number of layers)
        - args: Additional arguments (if needed)
        """
        super(UNET_2D_simp, self).__init__()
        self.input_channels = input_channels
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.spec_freq = spect_freq
        self.spec_time = spect_time

        # Pooling layers (2D average pooling)
        self.AvgPool2D1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.AvgPool2D2 = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))

        # Encoder
        self.layer1 = self.down_layer_2d(self.input_channels, self.layer_n, self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.layer2 = self.down_layer_2d(self.layer_n, self.layer_n * 2, self.kernel_size, stride=2, padding=self.kernel_size//2)
        # Concatenate with pooled input (similar to adding residual skip from input)
        self.layer3 = self.down_layer_2d(self.layer_n * 2 + self.input_channels, self.layer_n * 3, self.kernel_size, stride=2, padding=self.kernel_size//2)
        self.layer4 = self.down_layer_2d(self.layer_n * 3 + self.input_channels, self.layer_n * 4, self.kernel_size, stride=2, padding=self.kernel_size//2)

        # Decoder
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.cbr_up1 = conbr_block_2d(self.layer_n * 4 + self.layer_n * 3, self.layer_n * 3, self.kernel_size, stride=1, padding=self.kernel_size//2)
        # self.cbr_up2 = conbr_block_2d(self.layer_n * 3 + self.layer_n * 2, self.layer_n * 2, self.kernel_size, stride=1, padding=self.kernel_size//2)
        # self.cbr_up3 = conbr_block_2d(self.layer_n * 2 + self.layer_n, self.layer_n, self.kernel_size, stride=1, padding=self.kernel_size//2)

        # Final output convolution layer to map to desired output channels
        # self.outcov = nn.Conv2d(self.layer_n, output_channels, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)

        if self.spec_freq == 48 and self.spec_time == 200:
            self.fc = nn.Linear(25, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 128:
            self.fc = nn.Linear(16, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 100:
            self.fc = nn.Linear(13, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 480:
            self.fc = nn.Linear(60, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 1000:
            self.fc = nn.Linear(125, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 200:
            self.fc = nn.Linear(25, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 3004:
            self.fc = nn.Linear(376, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 3000:
            self.fc = nn.Linear(375, 1)
            self.fc2 = nn.Linear(6, 1)
    def down_layer_2d(self, in_channels, out_channels, kernel_size, stride, padding):
        """Creates a downsampling layer using a conv block."""
        return nn.Sequential(
            conbr_block_2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        # x is expected to have shape: (batch, channels, freq, time)
        # Create multi-scale pooled inputs from the original signal for skip connections.
        pool_x1 = self.AvgPool2D1(x)  # Downsample by factor of 2
        pool_x2 = self.AvgPool2D2(x)  # Downsample by factor of 4
        # Encoder path
        out_0 = self.layer1(x)          # (B, layer_n, F, T)
        out_1 = self.layer2(out_0)        # (B, layer_n*2, F/2, T/2)

        # Concatenate skip connection from original input (pooled) with out_1
        x1 = torch.cat([out_1, pool_x1], dim=1)  # (B, layer_n*2 + input_channels, F/2, T/2)
        out_2 = self.layer3(x1)          # (B, layer_n*3, F/4, T/4)

        x2 = torch.cat([out_2, pool_x2], dim=1)  # (B, layer_n*3 + input_channels, F/4, T/4)
        x3 = self.layer4(x2)             # (B, layer_n*4, F/8, T/8)

        # Decoder path
        # up = self.upsample(x3)           # Upsample to (B, layer_n*4, F/4, T/4)
        # up = torch.cat([up, out_2], dim=1)  # (B, layer_n*4 + layer_n*3, F/4, T/4)
        # up = self.cbr_up1(up)            # (B, layer_n*3, F/4, T/4)

        # up = self.upsample(up)           # Upsample to (B, layer_n*3, F/2, T/2)
        # up = torch.cat([up, out_1], dim=1)   # (B, layer_n*3 + layer_n*2, F/2, T/2)
        # up = self.cbr_up2(up)            # (B, layer_n*2, F/2, T/2)

        # up = self.upsample(up)           # Upsample to (B, layer_n*2, F, T)
        # up = torch.cat([up, out_0], dim=1)   # (B, layer_n*2 + layer_n, F, T)
        # up = self.cbr_up3(up)            # (B, layer_n, F, T)

        # out = self.outcov(up)            # (B, output_channels, F, T)

        return None, self.fc2(self.fc(x3).squeeze()).squeeze()


# In[ ]:


import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from .backbones import *
from .models_nc import *
from .TC import *
from utils import WaveletTransform, FourierTransform
from data_preprocess import augmentations

class SimCLR(nn.Module):
    def __init__(self, backbone, dim=128):
        super(SimCLR, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='SimCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

    def forward(self, x1, x2,  DACL_training=False):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, z1, z2
        else:
            return z1, z2

class NNCLR(nn.Module):
    def __init__(self, backbone, dim=128, pred_dim=64):
        super(NNCLR, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='NNCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.predictor = Predictor(model='NNCLR', dim=dim, pred_dim=pred_dim)

    def forward(self, x1, x2):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, p1, p2, z1.detach(), z2.detach()
        else:
            return p1, p2, z1.detach(), z2.detach()  

class BYOL(nn.Module):
    def __init__(
        self,
        DEVICE,
        backbone,
        window_size = 30,
        n_channels = 77,
        hidden_layer = -1,
        projection_size = 64,
        projection_hidden_size = 256,
        moving_average = 0.99,
        use_momentum = True,
    ):
        super().__init__()

        net = backbone
        self.bb_dim = net.out_dim
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, DEVICE=DEVICE, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average)

        self.online_predictor = Predictor(model='byol', dim=projection_size, pred_dim=projection_hidden_size)

        self.to(DEVICE)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, window_size, n_channels, device=DEVICE),
                     torch.randn(2, window_size, n_channels, device=DEVICE))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x1,
        x2,
        return_embedding = False,
        return_projection = True,
        require_lat = False
    ):
        assert not (self.training and x1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection = return_projection)

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            online_proj_one, x1_decoded, lat1 = self.online_encoder(x1)
            online_proj_two, x2_decoded, lat2 = self.online_encoder(x2)
        else:
            online_proj_one, lat1 = self.online_encoder(x1)
            online_proj_two, lat2 = self.online_encoder(x2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
                target_proj_one, _, _ = target_encoder(x1)
                target_proj_two, _, _ = target_encoder(x2)
            else:
                target_proj_one, _ = target_encoder(x1)
                target_proj_two, _ = target_encoder(x2)

            target_proj_one.detach_()
            target_proj_two.detach_()

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            if require_lat:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2
            else:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()
        else:
            if require_lat:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2
            else:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()

class TSTCC(nn.Module):
    def __init__(self, backbone, DEVICE, temp_unit='tsfm', tc_hidden=100):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(TSTCC, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_channels
        self.TC = TC(self.bb_dim, DEVICE, tc_hidden=tc_hidden, temp_unit=temp_unit).to(DEVICE)
        self.projector = Projector(model='TS-TCC', bb_dim=self.bb_dim, prev_dim=None, dim=tc_hidden)

    def forward(self, x1, x2, DACL_training=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        _, z1 = self.encoder(x1)
        _, z2 = self.encoder(x2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        nce1, c_t1 = self.TC(z1, z2)
        nce2, c_t2 = self.TC(z2, z1)

        p1 = self.projector(c_t1)
        p2 = self.projector(c_t2)

        return nce1, nce2, p1, p2

"""
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

"""
class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.p)

    def forward(self, x, y):
        repr_loss = torch.nn.functional.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

"""
https://github.com/facebookresearch/barlowtwins/blob/main/main.py

"""
class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(args.p, affine=False).to(args.cuda)        

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

class CLIP(nn.Module):
    def __init__(self, backbone, backbone_FT, DEVICE, dim=128, args=None):
        super(CLIP, self).__init__()
        self.encoder = backbone
        self.args = args
        self.encoder_FT = backbone_FT
        self.bb_dim = self.encoder.out_dim

        self.projector = Projector(model='CLIP', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.projector_FT = Projector(model='CLIP_FT', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.device = DEVICE

    def forward(self, x1, x2):
        x2 = x2.transpose(1, 2) 
        _, z1 = self.encoder(x1)
        z2 = self.encoder_FT(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector_FT(z2)

        return z1, z2

class MTM(nn.Module):
    def __init__(self, backbone, DEVICE, dim=128, args=None):
        super(MTM, self).__init__()
        self.encoder = backbone
        self.args = args
        self.bb_dim = self.encoder.out_dim 
        self.device = DEVICE
        self.projector = Projector(model='MTM', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

    def forward(self, x1, x2):
        data_masked_om = torch.cat([x1, x2], 0) # data_masked_om = torch.cat([data, data_masked_m], 0)
        _, h = self.encoder(data_masked_om)
        z = self.projector(h)
        return z, h, data_masked_om        

class IsoAlign(nn.Module):
    def __init__(self, backbone, spect_encoder, FT_encoder, DEVICE, dim=128, batch_size=1024, args=None):
        super(IsoAlign, self).__init__()
        self.encoder = backbone
        self.args = args
        self.batch_size = batch_size
        self.spect_encoder = spect_encoder
        self.FT_encoder = FT_encoder
        self.bb_dim = self.encoder.out_dim 
        self.device = DEVICE
        self.projector = Projector(model='IsoAlign', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.projector_spect = Projector(model='IsoAlign', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.projector_FT = Projector(model='IsoAlign', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

        # self.predictor_FT = nn.Linear(dim, dim) # Ablation
        self.predictor_FT = ConvMapping(in_channels=dim, hidden_channels=64, kernel_size=3).to(DEVICE)
        # self.predictor_FT = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)) # Ablation

        # self.predictor_cwt = nn.Linear(dim, dim) # Ablation
        self.predictor_cwt = ConvMapping(in_channels=dim, hidden_channels=64, kernel_size=3).to(DEVICE)
        # self.predictor_cwt = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)) # Ablation

        self.wavelet_transform = WaveletTransform(wavelet='cmor1-1', fs=25)
        self.FT_transform = FourierTransform(fs=25)

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.steps = 32

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)      

    def cont_loss(self, r1, r2, batch_size, sim_matrix=None):
        representations = torch.cat([r1, r2], dim=0)

        similarity_matrix = torch.matmul(representations, representations.t())

        # Adjust the top-left quadrant using the original similarity for time series
        if sim_matrix is not None:
            similarity_matrix[:batch_size, :batch_size] *= sim_matrix       

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= 0.15

        labels = torch.zeros(2 * batch_size).to(logits.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)

    def calc_loss(self, e1, e2, e3): # e1 -> Time, e2 -> CWT, e3 -> FT
        e1 = torch.nn.functional.normalize(e1, dim=1)
        e2 = torch.nn.functional.normalize(e2, dim=1)
        e3 = torch.nn.functional.normalize(e3, dim=1)

        predicted_FT_latent = self.predictor_FT(e1)
        predicted_cwt_latent = self.predictor_cwt(e1)

        loss_predicted_cwt = torch.nn.functional.l1_loss(predicted_cwt_latent, e2) / self.batch_size
        loss_predicted_FT = torch.nn.functional.l1_loss(predicted_FT_latent, e3) / self.batch_size

        # symmetric loss functions
        loss1 = self.cont_loss(e1, e2, self.batch_size) 
        loss2 = self.cont_loss(e1, e3, self.batch_size)
        loss3 = self.cont_loss(e2, e3, self.batch_size)

        # if self.args.wo_OB: # without ortohogonal base
        #     return loss1 + (loss_predicted_cwt)
        # elif self.args.wo_OF: # without overcomplete frames
        #     return loss2 + (loss_predicted_FT)
        # else: # usual
        #     return loss1 + loss2 + loss3 + (loss_predicted_FT + loss_predicted_cwt)

        return loss1 + loss2 + loss3 + (loss_predicted_FT + loss_predicted_cwt)

        # return loss1 + loss2 + loss3

    def forward(self, x1, x2, x3):
        B, C, T = x1.shape  # Original shape (B, C, T)

        x1 = x1.transpose(1, 2) # (B, C, T) -> (B, T, C)
        x2 = x2.permute(0, 3, 1, 2)  # (B, F, T, C) -> (B, C, F, T)
        # x3.shape = (B, C, F)

        _, R_t = self.encoder(x1)
        out, R_f = self.spect_encoder(x2)
        _, R_f_FT = self.FT_encoder(x3)

        R_t = self.projector(R_t)
        R_f = self.projector_spect(R_f)
        R_f_FT = self.projector_FT(R_f_FT)

        loss = self.calc_loss(R_t, R_f, R_f_FT)

        # Ablations

        # R_t = torch.nn.functional.normalize(R_t, dim=1)
        # R_f = torch.nn.functional.normalize(R_f, dim=1)
        # R_f_FT = torch.nn.functional.normalize(R_f_FT, dim=1)

        # predicted_FT_latent = self.predictor1(R_t)
        # predicted_cwt_latent = self.predictor2(R_t)

        # loss_predicted_FT = torch.nn.functional.l1_loss(predicted_FT_latent, R_f) 
        # loss_predicted_cwt = torch.nn.functional.l1_loss(predicted_cwt_latent, R_f_FT)

        # loss_consistency = torch.dist(self.predictor2.weight @ torch.linalg.inv(self.predictor1.weight), self.const.weight) / B

        # symmetric loss functions
        # loss1 = self.cont_loss(R_t, R_f, self.batch_size)
        # loss2 = self.cont_loss(R_t, R_f_FT, self.batch_size)
        # loss3 = self.cont_loss(R_f, R_f_FT, self.batch_size)

        # loss1_1 = torch.nn.functional.pairwise_distance(R_t, R_f, p=2).sum() / B
        # loss2_1 = torch.nn.functional.pairwise_distance(R_t, R_f_FT, p=2).sum() / B
        # loss3_1 = torch.nn.functional.pairwise_distance(R_f, R_f_FT, p=2).sum() / B

        # loss1 = self.cont_loss_negs(R_t, R_f_FT, self.batch_size, sim_matrix=sim_matrix_FT)
        # loss2 = self.cont_loss_negs(R_t, R_f, self.batch_size, sim_matrix=sim_matrix_FT)
        # loss3 = self.cont_loss_negs(R_f, R_f_FT, self.batch_size, sim_matrix=sim_matrix_FT)

        return loss
        # return loss1 + loss2 + loss3 + (loss_predicted_FT + loss_predicted_cwt)
        # return loss_predicted_FT + loss_predicted_cwt
        # return loss_global + loss1_1 + loss2_1 + loss3_1


class IntegratedEncoder(nn.Module):
    def __init__(self, model, map_ft, map_cwt, args, out_dim):
        super(IntegratedEncoder, self).__init__()
        self.encoder = model.encoder
        self.map_ft = map_ft  
        self.map_cwt = map_cwt 
        self.wo_OF = args.wo_OF
        self.wo_OB = args.wo_OB        
        self.out_dim = out_dim * 3 if not self.wo_OF and not self.wo_OB else out_dim * 2
        # self.out_dim = out_dim * 1 if not self.wo_OF and not self.wo_OB else out_dim * 2 # second ablation

    def forward(self, x_sample):
        _, time_emb = self.encoder(x_sample)

        # Map the embeddings using the learned mappers.
        ft_mapped  = self.map_ft(time_emb)
        cwt_mapped = self.map_cwt(time_emb)

        # Ablations
        # if self.wo_OF: # without overcomplete frames
        #     all_emb = torch.cat([ft_mapped, time_emb], dim=1)
        # elif self.wo_OB:
        #     all_emb = torch.cat([cwt_mapped, time_emb], dim=1)
        # else: # normal    
        #     all_emb = torch.cat([ft_mapped, time_emb, cwt_mapped], dim=1)

        all_emb = torch.cat([ft_mapped, time_emb, cwt_mapped], dim=1)

        # all_emb = time_emb

        return _, all_emb


# In[ ]:


# ### 1. Custom Raw Preprocessing Pipeline

# %%
import glob

print("Locating local raw recordings from Sleep-EDF...")

# 🚨 CHANGE THIS to the actual folder path where your local .edf files are stored
LOCAL_DATA_DIR = "/home/gella.saikrishna/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette"

# Automatically find all PSG and Hypnogram files in your directory
psg_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*PSG.edf")))
hypno_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*Hypnogram.edf")))

# Zip them together so they pair up correctly
raw_data_files = list(zip(psg_files, hypno_files))

if len(raw_data_files) == 0:
    raise FileNotFoundError(f"No EDF files found in {LOCAL_DATA_DIR}. Check your path!")

print(f"✅ Found {len(raw_data_files)} paired recording sessions locally.")

all_epochs_list = []
all_labels_list = []

print("\nPreprocessing raw EEG signals...")
for psg_file, hypno_file in raw_data_files:
    raw = mne.io.read_raw_edf(psg_file, preload=True)
    eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
    if len(eeg_channels) > 0:
        raw.pick_channels([eeg_channels[0]])
    else:
        continue

    raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin')
    if raw.info['sfreq'] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad="auto")

    annotations = mne.read_annotations(hypno_file)
    raw.set_annotations(annotations, emit_warning=False)

    events, event_id = mne.events_from_annotations(raw, event_id=STAGE_MAPPING, chunk_duration=float(EPOCH_SEC))
    tmax = 30.0 - 1.0 / TARGET_SFREQ 
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0.0, tmax=tmax, baseline=None, preload=True, reject=None)

    data = epochs.get_data() 
    labels = epochs.events[:, -1]
    data = np.squeeze(data, axis=1)

    all_epochs_list.append(data)
    all_labels_list.append(labels)

X = np.concatenate(all_epochs_list, axis=0)
y = np.concatenate(all_labels_list, axis=0)

print("Applying global Z-score standardization...")
X = (X - np.mean(X)) / (np.std(X) + 1e-6)

print("Splitting dataset into stratified subsets...")
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val)

# Packing the variables using identical keys ('samples', 'labels') expected by downstream components
processed_dataset_object = {
    "train": {"samples": torch.from_numpy(X_train).float(), "labels": torch.from_numpy(y_train).long()},
    "val": {"samples": torch.from_numpy(X_val).float(), "labels": torch.from_numpy(y_val).long()},
    "test": {"samples": torch.from_numpy(X_test).float(), "labels": torch.from_numpy(y_test).long()}
}
torch.save(processed_dataset_object, OUTPUT_FILE)
print(f"✅ Successfully created and saved custom splits to {OUTPUT_FILE}")


# In[ ]:


# %% [markdown]
# ### 2. Self-Supervised Multi-View Datasets

# %%
class SleepEDF_Pretrain_Dataset(Dataset):
    def __init__(self, pt_file_path=OUTPUT_FILE, split="train"):
        data_obj = torch.load(pt_file_path, map_location="cpu")[split]
        self.data = data_obj["samples"]
        self.window = torch.hann_window(128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_t = self.data[idx] 
        x_time = x_t.unsqueeze(0) 

        x_fft = torch.fft.rfft(x_t)
        x_fourier = torch.stack([torch.abs(x_fft), torch.angle(x_fft)], dim=0)

        x_stft = torch.stft(x_t, n_fft=128, hop_length=64, window=self.window, return_complex=True)

        # Exact dimension alignment matching your .ipynb file
        x_wavelet = torch.abs(x_stft)[:64, :] 
        x_wavelet = F.pad(x_wavelet, (0, 1)) 
        x_wavelet = x_wavelet.unsqueeze(0) 

        return x_time, x_fourier, x_wavelet

class SleepEDF_Evaluation_Dataset(Dataset):
    def __init__(self, pt_file_path=OUTPUT_FILE, split="train"):
        data_obj = torch.load(pt_file_path, map_location="cpu")[split]
        self.samples = data_obj["samples"].unsqueeze(2)  # Reshaped to [N, 3000, 1] for internal .transpose()
        self.labels = data_obj["labels"].long()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# In[ ]:


# %% [markdown]
# ### 3. Self-Supervised Pre-Training (IsoAlign)

# %%
pretrain_loader = DataLoader(SleepEDF_Pretrain_Dataset(split="train"), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Determine shapes dynamically matching the dataset outputs
_, _, sample_w = SleepEDF_Pretrain_Dataset(split="train")[0]
SPECT_FREQ = sample_w.shape[1]   
SPECT_TIME = sample_w.shape[2]   

# A. Time Encoder Architecture Configuration
time_encoder = ResNet1D(
    in_channels=1, base_filters=32, kernel_size=5, stride=1, groups=1, 
    n_block=3, n_classes=LATENT_DIM, downsample_gap=2, increasefilter_gap=4, 
    use_do=True, backbone=True, output_dim=LATENT_DIM
).to(DEVICE)

# B. Wavelet Spectrogram Encoder Architecture Configuration
spect_encoder = UNET_2D_simp(
    input_channels=1, output_channels=LATENT_DIM, layer_n=32, 
    spect_freq=SPECT_FREQ, spect_time=SPECT_TIME, kernel_size=3
)
# Monkey-patching identical layer dimensions from your notebook
spect_encoder.fc = nn.Linear(6, 1)
spect_encoder.fc2 = nn.Linear(8, 1)
spect_encoder = spect_encoder.to(DEVICE)

# C. Fourier Encoder Wrapper Architecture Configuration
class FourierWrapper(nn.Module):
    def __init__(self, in_length):
        super().__init__()
        self.enc = FourierEncoder(in_channels=2, in_length=in_length, out_channels=LATENT_DIM)
        if in_length == 3000:
            self.enc.fc_abs = nn.Linear(376, 1)
            self.enc.fc_angle = nn.Linear(376, 1)

    def forward(self, x): 
        return None, self.enc(x)

ft_encoder = FourierWrapper(in_length=TIME_STEPS).to(DEVICE)

# Framework Setup
class Args: wo_OB = False; wo_OF = False
model = IsoAlign(backbone=time_encoder, spect_encoder=spect_encoder, FT_encoder=ft_encoder, DEVICE=DEVICE, dim=LATENT_DIM, batch_size=BATCH_SIZE, args=Args()).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("\n🚀 Starting Self-Supervised Multi-View Pre-training...")
best_loss = float('inf')

for epoch in range(PRETRAIN_EPOCHS):
    model.train()
    total_loss = 0
    for batch_t, batch_f, batch_w in pretrain_loader:
        batch_t, batch_f = batch_t.to(DEVICE), batch_f.to(DEVICE)
        batch_w = batch_w.permute(0, 2, 3, 1).to(DEVICE) 

        optimizer.zero_grad()
        loss = model(batch_t, batch_w, batch_f)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(pretrain_loader)
    print(f"Pretrain Epoch [{epoch+1:02d}/{PRETRAIN_EPOCHS}] | Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "lwa_best_time_encoder.pth"))


# In[ ]:


# %% [markdown]
# ### 4. Downstream Evaluation (Frozen Pre-trained Backbone + 1-Layer Linear Head)

# %%
train_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="train"), batch_size=128, shuffle=True)
val_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="val"), batch_size=128, shuffle=False)
test_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="test"), batch_size=128, shuffle=False)

# Reinitialize backbone and load pre-trained weights
eval_encoder = ResNet1D(
    in_channels=1, base_filters=32, kernel_size=5, stride=1, groups=1, 
    n_block=3, n_classes=LATENT_DIM, downsample_gap=2, increasefilter_gap=4, 
    use_do=False, backbone=True, output_dim=LATENT_DIM
).to(DEVICE)

eval_encoder.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "lwa_best_time_encoder.pth"), map_location=DEVICE))
print("\n🔒 Loaded Pre-trained Encoder weights and freezing parameters...")

for param in eval_encoder.parameters():
    param.requires_grad = False

# Exact 1-Layer Linear head setup configuration from your notebook block (nn.Linear(128, 5))
classifier_head = nn.Linear(LATENT_DIM, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
eval_optimizer = optim.Adam(classifier_head.parameters(), lr=1e-2, weight_decay=1e-4)

# %%
def evaluate_pipeline(encoder_m, head_m, loader, print_breakdown=False):
    encoder_m.eval(); head_m.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for signals, labels in loader:
            _, features = encoder_m(signals.to(DEVICE))
            preds = torch.argmax(head_m(features), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_targets, all_preds)

    if print_breakdown:
        CLASS_NAMES = ["Wake (W)", "N1 Stage", "N2 Stage", "N3 Stage", "REM"]
        print("\n📊 DETAILED PERFORMANCE BREAKDOWN (PRE-TRAINED BACKBONE + 1-LAYER LINEAR HEAD):")
        print(f"   ➡️ Test Accuracy:    {acc*100:.2f}%")
        print(f"   ➡️ Cohen's Kappa:     {kappa:.4f}")
        print(f"   ➡️ Macro F1-Score:    {macro_f1:.4f}")
        print(f"   ➡️ Weighted F1-Score: {weighted_f1:.4f}")
        print("   ➡️ Stage-Specific F1-Scores:")
        for name, score in zip(CLASS_NAMES, f1_score(all_targets, all_preds, average=None)):
            print(f"       • {name.ljust(12)}: {score:.4f}")
    return acc, macro_f1

# %%
print("\n🏋️ Training Supervised 1-Layer Linear Head...")
best_val_f1 = 0.0

for epoch in range(EVAL_EPOCHS):
    eval_encoder.eval(); classifier_head.train()
    total_loss = 0
    for signals, labels in train_loader:
        eval_optimizer.zero_grad()
        with torch.no_grad():
            _, features = eval_encoder(signals.to(DEVICE))
        loss = criterion(classifier_head(features), labels.to(DEVICE))
        loss.backward()
        eval_optimizer.step()
        total_loss += loss.item()

    val_acc, val_f1 = evaluate_pipeline(eval_encoder, classifier_head, val_loader)
    print(f"Eval Epoch [{epoch+1:02d}/{EVAL_EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(classifier_head.state_dict(), os.path.join(CHECKPOINT_DIR, "linear_classifier_head.pth"))


# In[ ]:


# %% [markdown]
# ### 5. Final Report Verification

# %%
print("\n🔒 Final Deployment Testing on Unseen Participants...")
classifier_head.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "linear_classifier_head.pth"), map_location=DEVICE))
_, _ = evaluate_pipeline(eval_encoder, classifier_head, test_loader, print_breakdown=True)

