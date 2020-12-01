# from mmdet.ops.carafe import CARAFEPack     #mmdetection
from mmcv.ops.carafe import CARAFEPack
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack

import torch
from torch import nn
import torch.nn.functional as F
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# from models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from models.ShuffleNetV2 import shufflenet_v2_x1_0

# from models.res2net import res2net50_26w_6s,res2net50_26w_4s
# from models.res2net_dla import res2net_dla60, res2next_dla60

#from models.dcn_resnet import load_dcn_resnet

#from utils.utils import show_feature_map
from utils import show_feature_map
import math
import config

def load_dcn_resnet():
    pass

# from models.encoding.danet_head import DANetHead

d = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
     'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
     'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
     'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
     'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
     # 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
     # 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
     'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]},
     # 'res2net50_26w_6s': {'models': res2net50_26w_6s, 'out': [256, 512, 1024, 2048]},
     # 'res2net_dla60': {'models': res2net_dla60, 'out': [128, 256, 512, 1024]},
     # 'dcn_resnet50': {'models': load_dcn_resnet, 'out': [256, 512, 1024, 2048]}
     }
inplace = True


class FPN_ResNet_CARAFE(nn.Module):
    def __init__(self, backbone, result_num, scale: int = 1, pretrained=False, predict=False):
        super(FPN_ResNet_CARAFE, self).__init__()
        assert backbone in d, 'backbone must in: {}'.format(d)
        self.scale = scale
        conv_out = 256
        model, out = d[backbone]['models'], d[backbone]['out']
        self.backbone_name = backbone
        self.backbone = model(pretrained=pretrained)
        self.predict = predict

        # Top layer
        self.toplayer = nn.Conv2d(out[3], conv_out, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(out[2], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(out[1], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(out[0], conv_out, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        # self.smooth1 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(conv_out * 4, conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=inplace)
        )

        #self.DANetHead = DANetHead(conv_out, result_num, norm_layer=nn.BatchNorm2d)

        # self.out_up_CARAFE =  CARAFEPack(conv_out, 2, up_kernel=5, compressed_channels=64)
        self.deform = ModulatedDeformConv2dPack(conv_out,conv_out,3,1,1)
        self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input: torch.Tensor):
        _, _, H, W = input.size()

        if self.backbone_name == 'dcn_resnet50':
            res_dict = self.backbone(input)
            c2, c3, c4, c5 = res_dict['res2'], res_dict['res3'], res_dict['res4'], res_dict['res5']
        else:
            c2, c3, c4, c5 = self.backbone(input)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        # p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        # x = self.out_up_CARAFE(x)   #上采样2倍
        x = self.deform(x)
        x = self.out_conv(x)

        if self.train:
            if config.scale_model == 'nearest':
                x = F.interpolate(x, size=(H, W), mode=config.scale_model)
            else:
                x = F.interpolate(x, size=(H, W), mode=config.scale_model, align_corners=True)
        else:
            if config.scale_model == 'nearest':
                x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode=config.scale_model)
            else:
                x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode=config.scale_model, align_corners=True)
        if self.predict:
            show_feature_map([c2, c3, c4, c5, p2, p3, p4, p5, x], row=3, column=4)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)



if __name__ == '__main__':
    import time

    device = torch.device('cpu')  #cuda:0
    backbone = 'resnet50'
    net = FPN_ResNet_CARAFE(backbone=backbone, pretrained=False, result_num=2, predict=False).to(device)
    net.eval()
    x = torch.randn(1, 3, 256, 256).to(device)
    start = time.time()
    y = net(x)
    print(time.time() - start)  # 18->4.5  50->5.8
    # print(y.shape)   #torch.Size([1, 5, 512, 512])
    # # torch.save(net.state_dict(),f'{backbone}.pth')

    from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

    print_model_parm_flops(net, x)
    print_model_parm_nums(net)
    #show_summary(net, 'E:/summery.xlsx')
