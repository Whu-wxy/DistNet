import torch
from torch import nn
import torch.nn.functional as F
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

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
     'dcn_resnet50': {'models': load_dcn_resnet, 'out': [256, 512, 1024, 2048]}
     }
inplace = True


class SA_FPN(nn.Module):
    def __init__(self, backbone, result_num, scale: int = 1, pretrained=False, predict=False):
        super(SA_FPN, self).__init__()
        assert backbone in d, 'backbone must in: {}'.format(d)
        self.scale = scale
        conv_out = 256
        model, out = d[backbone]['models'], d[backbone]['out']
        self.backbone_name = backbone
        self.backbone = model(pretrained=pretrained)
        self.predict = predict

        # Top layer
        self.toplayer = nn.Conv2d(out[3], conv_out, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.top_bn = nn.BatchNorm2d(conv_out)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(out[2], conv_out, kernel_size=1, stride=1, padding=0)
        self.lat_bn1 = nn.BatchNorm2d(conv_out)
        self.latlayer2 = nn.Conv2d(out[1], conv_out, kernel_size=1, stride=1, padding=0)
        self.lat_bn2 = nn.BatchNorm2d(conv_out)
        self.latlayer3 = nn.Conv2d(out[0], conv_out, kernel_size=1, stride=1, padding=0)
        self.lat_bn3 = nn.BatchNorm2d(conv_out)

        self.up_conv1 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        self.up_bn1 = nn.BatchNorm2d(conv_out)
        self.up_conv2 = nn.Conv2d(conv_out, conv_out, kernel_size=1, stride=1, padding=0)
        self.up_bn2 = nn.BatchNorm2d(conv_out)
        #### Add Upsamle
        self.up_conv3 = nn.Conv2d(conv_out, 128, kernel_size=3, stride=1, padding=1)
        self.up_conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
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
        p5 = self.top_bn(p5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.lat_bn1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.lat_bn2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.lat_bn3(p2)

        x = self.up_conv1(p2)
        x = self.up_bn1(x)
        x = self.up_conv2(x)
        x = self.up_bn2(x)

        x = F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True)
        x = self.up_conv3(x)
        x = self.up_conv4(x)

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
    net = SA_FPN(backbone=backbone, pretrained=False, result_num=1, predict=False).to(device)
    net.eval()
    x = torch.randn(1, 3, 512, 512).to(device)
    start = time.time()
    y = net(x)
    print(time.time() - start)  # 18->4.5  50->5.8
    print(y.shape)   #torch.Size([1, 5, 512, 512])
    # # torch.save(net.state_dict(),f'{backbone}.pth')

    from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

    print_model_parm_flops(net, x)
    print_model_parm_nums(net)
    #show_summary(net, 'E:/summery.xlsx')
