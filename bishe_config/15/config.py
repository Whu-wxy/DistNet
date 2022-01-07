name = 'distv2_total_exdata'

# data config
dataset_type = 'ctw1500'    # ctw1500  total 在train_ic15.py和在train_ic17.py中不适用这个参数

trainroot = '../data/IC15/train'
testroot = '../data/IC15/test'
output_dir = '../.save/IC15/dla3'
eval_script = 'iou'   # deteval, iou, 2013
data_shape = 640    # 640

long_size = 1600  # 2240/None
img_norm = False
augment_list = ['flip', 'rotate', 'resize']   # ['flip', 'rotate', 'resize', 'rotate90']
random_scales = [0.5, 1, 2.0, 3.0]    #[0.5, 1, 2.0, 3.0]
uniform_scales = False
cp=False
elastic = False

# train config
gpu_id = '0'
workers = 10
pin_memory = True
start_epoch = 0
epochs = 250   #600
early_stop=20  #test F1

train_batch_size = 14
try_test_epoch = [25, 50, 75, 100, 115, 125, 150, 175, 200, 225, 249]
start_test_epoch = 150      #绝对值
test_inteval = 3
always_test_threld = 0.68

test_for_loss_inteval = 5
dla_model = True

# Learning rate
optim = 'ranger'   #  sgd/adam/adamw/radam/ranger/adabound
weight_decay = 5e-4    #5e-4
amsgrad = False

lr = 1e-3
end_lr = 1e-7


lr_scheduler='MultiStepLR'
if lr_scheduler=='MultiStepLR':
    #MultiStepLR
    lr_gamma = 0.1     # 0.1
    lr_decay_step = [10]    #  [10, 100]
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
display_input_images = True
display_output_images = True
show_images_interval = 5000  #显示结果图片的iter间隔


# check points
pretrained = True   #backbone
restart_training = True # begin from 0 epoch
load_lr = False
checkpoint = ''   #full model ckpt
#../save/dist_IC17_3/DistNet_IC17_150_loss1.043292.pth
if len(checkpoint) != 0:
    pretrained = False


# net config
backbone = 'resnet50'
n = 1            # result_num
m = 0.5
min_threld = 0.3    #选出大图   0.2
max_threld = 0.7     #选出小图   0.7

bd_loss = False
bd_clip = False      ###################
clip_value = 50

OHEM_ratio = 3
scale = 4
scale_model = 'bilinear'
# mode:   'nearest' | 'linear'(3D) | 'bilinear' | 'bicubic' | 'trilinear'(5D) | 'area'
#align_corners:None,   true,          true,          true,      true,            None
seed = 2
decode_threld = 0.7311    # 0.58
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


