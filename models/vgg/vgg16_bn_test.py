from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True, mdConv=[]):
        super(vgg16_bn, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(13):
            if isinstance(vgg_pretrained_features[x], nn.Conv2d):
                conv = ModulatedDeformConv2dPack(vgg_pretrained_features[x].in_channels,
                                          vgg_pretrained_features[x].out_channels,
                                          vgg_pretrained_features[x].kernel_size,
                                          vgg_pretrained_features[x].stride,
                                          vgg_pretrained_features[x].padding,
                                          vgg_pretrained_features[x].dilation,
                                          vgg_pretrained_features[x].groups)
                                          #bias=vgg_pretrained_features[x].bias)
                self.slice1.add_module(str(x), conv)
            else:
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 23):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 33):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(33, 40):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(40, 43):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        # self.slice5 = torch.nn.Sequential(
        #         nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #         nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        #         nn.Conv2d(1024, 1024, kernel_size=1)
        # )
        # init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7


        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
            init_weights(self.slice5.modules())


        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


if __name__ == '__main__':
    import  time

    device = torch.device('cuda:0')   # cpu
    model = vgg16_bn(pretrained=False).to(device)
    model.eval()
    start = time.time()
    data = torch.randn(1, 3, 256, 256).to(device)
    output = model(data)
    print(time.time() - start)

    for out in output:
        print(out.shape)

    # 1.8951084613800049
    # torch.Size([1, 1024, 16, 16])
    # torch.Size([1, 512, 16, 16])
    # torch.Size([1, 512, 32, 32])
    # torch.Size([1, 256, 64, 64])
    # torch.Size([1, 128, 128, 128])
    # 20.98G
    # 18.13M

    # 0.3554525375366211
    # torch.Size([1, 512, 16, 16])
    # torch.Size([1, 512, 16, 16])
    # torch.Size([1, 512, 32, 32])
    # torch.Size([1, 256, 64, 64])
    # torch.Size([1, 128, 128, 128])
    # 5.78G
    # 14.79M


    from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

    print_model_parm_flops(model, data)
    print_model_parm_nums(model)

    show_summary(model, input_shape=(3, 256, 256), save_path='../../vgg16_bn_test.xlsx')

    # net = models.vgg16_bn(pretrained=False)
    # show_summary(net, input_shape=(3, 256, 256), save_path='E:/vgg16_bn_fix.xlsx')