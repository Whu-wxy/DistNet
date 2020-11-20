"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scnet import scnet50, scnet50_v1d, scnet101, scnet101_v1d
from models.vgg.vgg16_bn import init_weights
import math
import config

d = {'scnet50': {'models': scnet50, 'out': [256, 512, 1024, 2048]},
     'scnet50_v1d': {'models': scnet50_v1d, 'out': [256, 512, 1024, 2048]},
     'scnet101': {'models': scnet101, 'out': [256, 512, 1024, 2048]},
     'scnet101_v1d': {'models': scnet101_v1d, 'out': [256, 512, 1024, 2048]}
     }
inplace = True

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT_SC(nn.Module):
    def __init__(self, backbone='scnet50_v1d', num_out=1, pretrained=False, freeze=False, scale=1):
        super(CRAFT_SC, self).__init__()

        """ Base network """
        model, out = d[backbone]['models'], d[backbone]['out']
        self.backbone_name = backbone
        self.backbone = model(pretrained=pretrained)

        """ U network """
        self.upconv1 = double_conv(1024, 1024, 512)
        self.upconv2 = double_conv(1024, 512, 256)
        self.upconv3 = double_conv(512, 256, 128)
        self.upconv4 = double_conv(256, 128, 64)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_out, kernel_size=1),
        )

        self.scale = scale

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        _, _, H, W = x.size()

        """ Base network """
        sources = self.backbone(x)

        """ U network """
        y = torch.cat([sources[3]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[1]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[0].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[0]], dim=1)

        y = self.upconv4(y)

        y = self.conv_cls(y)

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
    model = CRAFT_SC(num_out=2, pretrained=False).to(device)
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

# 0.8840503692626953
# torch.Size([1, 2, 256, 256])
#   + Number of FLOPs: 8.50G
#   + Number of params: 33.03M

