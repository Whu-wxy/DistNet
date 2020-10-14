import torch
import torch.nn as nn

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

import logging
import os
import torch
from torch import nn
from torchocr.networks.CommonModules import ConvBNACT, SEBlock


class ResidualUnit(nn.Module):
    def __init__(self, num_in_filter, num_mid_filter, num_out_filter, stride, kernel_size, act=None, use_se=False):
        super().__init__()
        self.conv0 = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                               padding=0, act=act)

        self.conv1 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter, kernel_size=kernel_size,
                               stride=stride,
                               padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter, out_channels=num_mid_filter)
        else:
            self.se = None

        self.conv2 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                               padding=0)
        self.not_add = num_in_filter != num_out_filter or stride != 1

    def load_3rd_state_dict(self, _3rd_name, _state, _convolution_index):
        if _3rd_name == 'paddle':
            self.conv0.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_expand')
            self.conv1.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_depthwise')
            if self.se is not None:
                self.se.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_se')
            self.conv2.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_linear')
        else:
            pass
        pass

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y)
        if not self.not_add:
            y = x + y
        return y


class MobileNetV3(nn.Module):
    def __init__(self, in_channels, pretrained=True, **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super().__init__()
        self.scale = kwargs.get('scale', 0.5)
        model_name = kwargs.get('model_name', 'large')
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, self.scale)

        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        # conv1
        self.conv1 = ConvBNACT(in_channels=in_channels,
                               out_channels=self.make_divisible(inplanes * scale),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               act='hard_swish')
        i = 0
        inplanes = self.make_divisible(inplanes * scale)
        self.stages = nn.ModuleList()
        block_list = []
        self.out_channels = []
        for layer_cfg in cfg:
            if layer_cfg[5] == 2 and i > 2:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block = ResidualUnit(num_in_filter=inplanes,
                                 num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                                 num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                                 act=layer_cfg[4],
                                 stride=layer_cfg[5],
                                 kernel_size=layer_cfg[0],
                                 use_se=layer_cfg[3])
            block_list.append(block)
            inplanes = self.make_divisible(scale * layer_cfg[2])
            i += 1
        self.stages.append(nn.Sequential(*block_list))
        self.conv2 = ConvBNACT(
            in_channels=inplanes,
            out_channels=self.make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act='hard_swish')
        self.out_channels.append(self.make_divisible(scale * cls_ch_squeeze))

        if pretrained:
            ckpt_path = f'./weights/MobileNetV3_{model_name}_x{str(scale).replace(".", "_")}.pth'
            logger = logging.getLogger('torchocr')
            if os.path.exists(ckpt_path):
                logger.info('load imagenet weights')
                self.load_state_dict(torch.load(ckpt_path))
            else:
                logger.info(f'{ckpt_path} not exists')

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def load_3rd_state_dict(self, _3rd_name, _state):
        if _3rd_name == 'paddle':
            self.conv1.load_3rd_state_dict(_3rd_name, _state, 'conv1')
            m_block_index = 2
            for m_stage in self.stages:
                for m_block in m_stage:
                    m_block.load_3rd_state_dict(_3rd_name, _state, m_block_index)
                    m_block_index += 1
            self.conv2.load_3rd_state_dict(_3rd_name, _state, 'conv_last')
        else:
            pass

    def forward(self, x):
        x = self.conv1(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        out[-1] = self.conv2(out[-1])
        return out

	def _load_pretrained_model(self, model_url):
		pretrain_dict = model_zoo.load_url(model_url)
		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)
		logger.info('load pretrained models from imagenet')


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
	"""
	Constructs a MobileNetV2 architecture from
	`"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	model = MobileNetV2(**kwargs)
	if pretrained:
		model._load_pretrained_model[model_urls['mobilenet_v2']]
	return model

if __name__ == '__main__':
	import time

	device = torch.device('cpu')
	model = mobilenet_v2(pretrained=False).to(device)
	model.eval()
	fea = model.features

	start = time.time()
	data = torch.randn(1, 3, 256, 256).to(device)
	output = model(data)
	print(time.time() - start)
	print(output.shape)

	from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

	print_model_parm_flops(model, data)
	print_model_parm_nums(model)

