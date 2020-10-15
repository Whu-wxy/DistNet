"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg.vgg16_bn import vgg16_bn, init_weights
from models.ShuffleNetV2 import shufflenet_v2_x1_0

import config

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


class CRAFT(nn.Module):
    def __init__(self, num_out=1, pretrained=False, freeze=False, scale=1):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
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
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)

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
                y = F.interpolate(y, size=(H // self.scale, W // self.scale), mode=config.scale_model, align_corners=True)

        return y

if __name__ == '__main__':
    import  time

    device = torch.device('cpu')
    model = CRAFT(num_out=2, pretrained=False).to(device)
    model.eval()
    start = time.time()
    data = torch.randn(1, 3, 512, 512).to(device)
    output = model(data)
    print(time.time() - start)
    print(output.shape)

    from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

    print_model_parm_flops(model, data)
    print_model_parm_nums(model)

    #show_summary(model, input_shape=(3, 256, 256), save_path='E:/summery.xlsx')

# 1.2210698127746582
# torch.Size([1, 2, 256, 256])
#   + Number of FLOPs: 23.39G
#   + Number of params: 20.77M

# 4.9732842445373535
# torch.Size([1, 2, 512, 512])
#   + Number of FLOPs: 93.55G
#   + Number of params: 20.77M