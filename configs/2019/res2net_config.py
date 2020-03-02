# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun

# data config
trainroot = '../data/IC15/train'
testroot = '../data/IC15/test'
output_dir = '../save/CV/res2net50_26w_6s_2.1'
save_4_pt_box = True
eval_script = 'iou'   # deteval, iou, 2013
data_shape = 640    # 640

long_size = None  # 2240/None
img_norm = False
augment_list = ['flip', 'rotate', 'resize']   # ['flip', 'rotate', 'resize']
random_scales = [0.5, 1, 2.0, 3.0]    #[0.5, 1, 2.0, 3.0]
uniform_scales = False

# train config
gpu_id = '0'
workers = 3
start_epoch = 0
epochs = 700   #600
early_stop=10  #test F1

train_batch_size = 4
try_test_epoch = [1, 50, 100, 150, 200, 250, 300, 350, 360, 370, 380, 390, 400]
start_test_epoch = 500   #200    #相对值
test_inteval = 1
always_test_threld = 0.66

# Learning rate
optim = 'ranger'   #  sgd/adam/adamw/radam/ranger/adabound
weight_decay = 5e-4    #5e-4
amsgrad = False

lr = 1e-5
end_lr = 1e-7


lr_scheduler='MultiStepLR'
if lr_scheduler=='MultiStepLR':
    #MultiStepLR
    lr_gamma = 0.1
    lr_decay_step = [400]    #[250, 400]
elif lr_scheduler=='CyclicLR':
    #CyclicLR
    max_lr = 6e-5
    lr_gamma = 1.0
    step_size_up=1000
    LR_mode='triangular'           #{triangular, triangular2, exp_range}

if_warm_up = False
if optim=='radam' or optim=='ranger':
    if_warm_up = False
if if_warm_up:
    lr_gamma = 0.1
    warm_up_epoch = 6
    warm_up_lr = lr * lr_gamma
    lr_decay_step = [200, 400]

# Log
display_input_images = False
display_output_images = False
display_interval = 10     #print信息的batch间隔
show_images_interval = 50  #显示结果图片的batch间隔

# check points
pretrained = False   #backbone
restart_training = False   # begin from 0 epoch
checkpoint = '../save/CV/res2net50_26w_6s_2.0/Best_510_r0.765046_p0.848372_f10.804557.pth'
load_lr = False

# net config
backbone = 'res2net50_26w_6s'
Lambda = 0.7
n = 6            # 6
m = 0.5
OHEM_ratio = 3
scale = 1
scale_model = 'nearest'
# mode:   'nearest' | 'linear'(3D) | 'bilinear' | 'bicubic' | 'trilinear'(5D) | 'area'
#align_corners:None,   true,          true,          true,      true,            None
# random seed
seed = 2
decode_threld = 0.58    # 0.58
origin_shrink = True


def print():
    from pprint import pformat
    import json
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    with open(output_dir+'/config.json', 'w') as f:
        json.dump(tem_d, f, indent=4)
    return pformat(tem_d)


