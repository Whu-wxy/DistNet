"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg.vgg16_bn import vgg16_bn, init_weights
# from models.ShuffleNetV2 import shufflenet_v2_x1_0
from models.fapn import FaPNHead

import config

class FaPN_VGG16_bn(nn.Module):
    def __init__(self, num_out=1, pretrained=False, freeze=False, scale=1):
        super(FaPN_VGG16_bn, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        self.scale = scale
        temp_channels = 128
        # temp_channels = [32, 128, 128, 128, 256]
        temp_channels = [16, 64, 128, 128, 256]   # 2      26.28   23
        # temp_channels = [16, 64, 128, 256, 512]   # 2.45    27     24

        """ U network """
        self.head = FaPNHead([128, 256, 512, 512, 1024], temp_channels, num_out)

        # self.conv_cls = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(16, num_out, kernel_size=1),
        # )
        # init_weights(self.conv_cls.modules())

    def forward(self, x):
        _, _, H, W = x.size()

        """ Base network """
        features = self.basenet(x)
        features = features[:5]
        features = features[::-1]

        """ U network """
        y = self.head(features)

        # y = self.conv_cls(y)

        if self.training:
            if config.scale_model == 'nearest':
                y = F.interpolate(y, size=(H, W), mode=config.scale_model)
            else:
                y = F.interpolate(y, size=(H, W), mode=config.scale_model, align_corners=True)
        else:
            if config.scale_model == 'nearest':
                y = F.interpolate(y, size=(H // self.scale, W // self.scale), mode=config.scale_model)
            else:
                y = F.interpolate(y, size=(H // self.scale, W // self.scale), mode=config.scale_model,
                                  align_corners=True)

        return y


if __name__ == '__main__':
    import time

    device = torch.device('cpu')
    model = FaPN_VGG16_bn(num_out=2, pretrained=False).to(device)
    model.eval()
    start = time.time()
    data = torch.randn(1, 3, 256, 256).to(device)
    output = model(data)
    print(time.time() - start)
    print(output.shape)

    from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

    print_model_parm_flops(model, data)
    print_model_parm_nums(model)

    # show_summary(model, input_shape=(3, 256, 256), save_path='E:/summery.xlsx')
