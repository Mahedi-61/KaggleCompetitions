import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.utils import save_image

def conv_block(in_channels, out_channels, kernel_size, stride, padding, use_leaky, down, **kwargs):
  if down:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                     stride, padding, padding_mode="reflect", **kwargs)
  else:
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                              stride, padding, **kwargs)

  norm = nn.InstanceNorm2d(out_channels)

  if use_leaky:
    act = nn.LeakyReLU(0.2, inplace=True)
  else:
    act = nn.ReLU(inplace=True)

  return nn.Sequential(conv, norm, act)



class ResBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()

    layers = [
              conv_block(channels, channels, 3, 1, 1, False, True),
              nn.Conv2d(channels, channels, 3, 1, 1,  padding_mode="reflect"),
              nn.InstanceNorm2d(channels)
    ]

    self.res = nn.Sequential(*layers)
  
  def forward(self, x):
    return x + self.res(x)



class Generator(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, num_res_blocks=9):
    super().__init__()
    self.main = nn.Sequential(
        conv_block(in_channels, 64, 7, 1, padding=3, use_leaky=False, down=True), #64x256x256

        # down sample
        conv_block(64, 128, 3, 2, padding=1, use_leaky=False, down=True), #128x128x128
        conv_block(128, 256, 3, 2, padding=1, use_leaky=False, down=True), #256x64x64

        *[ResBlock(256) for _ in range(num_res_blocks)],

        # up sample
        conv_block(256, 128, 3, 2, padding=1, use_leaky=False, down=False, output_padding=1), #128x128x128
        conv_block(128, 64, 3, 2, padding=1, use_leaky=False, down=False, output_padding=1), #64x256x256
        nn.Conv2d(64, 3, 7, 1, padding=3, padding_mode="reflect"),
        nn.Tanh()
    )

  def forward(self, x):
    return self.main(x)



class Discriminator(nn.Module):
  def __init__(self, features=64):
    super().__init__()

    self.main = nn.Sequential(
        nn.Conv2d(3, features, 4, 2, padding=1, padding_mode="reflect"), #64x128x128
        nn.LeakyReLU(0.2, inplace=True),

        conv_block(features, features*2, 4, 2, padding=1, use_leaky=True, down=True), #128x64x64
        conv_block(features*2, features*4, 4, 2, padding=1, use_leaky=True, down=True), #256x32x32
        conv_block(features*4, features*8, 4, 1, padding=1, use_leaky=True, down=True), #512x31x31

        nn.Conv2d(features*8, 1, 4, padding=1, padding_mode="reflect"),  #1x30x30
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.main(x)