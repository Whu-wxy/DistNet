"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mobilenetv3 import mobilenetv3_large

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


inplace = True

class mobilenetv3_fpn(nn.Module):
    def __init__(self, num_out=1, model_path=None, freeze=False, scale=1):
        super(mobilenetv3_fpn, self).__init__()

        """ Base network """
        self.basenet = mobilenetv3_large(model_path=model_path)
        in_channels = self.basenet.out_channels
        conv_out = 256
        self.conv_out = conv_out
        out_channels = conv_out // 4

        """ FPN """
        # Top layer
        self.toplayer = nn.Conv2d(in_channels[3], out_channels, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )

        # self.up_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=self.conv_out, out_channels=self.conv_out // 4, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(self.conv_out // 4),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=self.conv_out // 4, out_channels=self.conv_out // 4, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(self.conv_out // 4),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=self.conv_out // 4, out_channels=num_out, kernel_size=2, stride=2),
        # )

        self.up_conv = nn.Conv2d(self.conv_out, 64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(64, num_out, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.scale = scale

    def forward(self, x):
        _, _, H, W = x.size()

        """ Base network """
        c2, c3, c4, c5 = self.basenet(x)

        """ U network """
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)

        #y = self.up_conv(x)

        x = self.up_conv(x)
        x = F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
        y = self.out_conv(x)

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

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)

if __name__ == '__main__':
    import  time

    device = torch.device('cpu')
    model = mobilenetv3_fpn(num_out=2, model_path='F:\\MobileNetV3_large_x0_5.pth').to(device)
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


# 0.2500143051147461
# torch.Size([1, 2, 256, 256])
#   + Number of FLOPs: 3.32G
#   + Number of params: 1.66M

# 0.9060518741607666
# torch.Size([1, 2, 512, 512])
#   + Number of FLOPs: 13.29G
#   + Number of params: 1.66M

################################
# 0.25101423263549805
# torch.Size([1, 2, 256, 256])
#   + Number of FLOPs: 3.32G
#   + Number of params: 1.68M

# 0.9370534420013428
# torch.Size([1, 2, 512, 512])
#   + Number of FLOPs: 13.29G
#   + Number of params: 1.68M