# -*- coding: utf-8 -*-
# @Time    : 1/4/19 11:18 AM
# @Author  : zhoujun
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import math


def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


def draw_bbox(img_path, result, color=(255, 0, 0),thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        if len(point) == 4:
            cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
            cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
            cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
            cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
        elif len(point) == 2:
            cv2.rectangle(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
    return img_path


def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


def save_checkpoint(checkpoint_path, model, optimizer, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    logger.info('models saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, logger, device, optimizer=None):
    state = torch.load(checkpoint_path, map_location=device)
    pretrained_dict = {k: v for k, v in state.items()}
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        # move_optimizer_to_cuda(optimizer)
    start_epoch = pretrained_dict['epoch']
    print('start epoch:', start_epoch)
    logger.info('start epoch: ' + str(start_epoch))
    logger.info('models loaded from %s' % checkpoint_path)
    return start_epoch

def move_optimizer_to_cuda(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if torch.is_tensor(param_state[k]):
                        param_state[k] = param_state[k].cuda(device=param.get_device())

# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back
    return newFunc


def show_feature_map(imgs, row=2, column=4, color=False):   #-5-->-2 save 256 picture?
    plt.figure()
    for i, img in enumerate(imgs):
        #print(img.shape)
        if len(img.shape) == 4:
            img = img.squeeze(dim=0)

        img = img.permute(1, 2, 0)
        # img = img[:, :, 0]
        img = torch.sum(img, dim=-1)
        #img = F.sigmoid(img)
        img = torch.sigmoid(img)

        img = img.detach().numpy()

        # to [0,255]
        img = np.round(img * 255)

        if column != None:
            plt.subplot(math.ceil(len(imgs)/column), column, i + 1)
        else:
            plt.subplot(row, math.ceil(len(imgs)/row), i + 1)
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def print_model_param(model_path):
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
    dict_name = list(model_dict)
    for i, p in enumerate(dict_name):
        print(i, p)


def load_part_checkpoint(checkpoint_path, model, device=torch.device('cuda:0'), part_id_list=[(0, -1)]):
    """
    :param checkpoint_path:
    :param model:
    :param part_id_list: [(0,0)] / [(0, 30), (60, 90)]
    :return:
    """
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state['state_dict']
    temp_dict = {}
    model_dict = model.state_dict()
    # for i, key in enumerate(model_dict.keys()):
    #     print(i , key)
    for i, key in enumerate(state_dict.keys()):
        if key.startswith('backbone'):
            key2 = key[9:]
        else:
            key2 = key
        for (min, max) in part_id_list:
            if max == -1:
                max = len(state_dict.keys())
            if i <= max and i >= min:
                temp_dict[key2] = state_dict[key]
    # for i, key in enumerate(temp_dict.keys()):
    #     print(i , key)

    model_dict.update(temp_dict)
    # for i, key in enumerate(model_dict.keys()):
    #     print(i , key)
    model.load_state_dict(model_dict)
    print('load part models params')


def torch_export(model, save_path):
    model.eval()
    data = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, data)
    traced_script_module.save(save_path)
    print("export finish.")


def torch_inport(save_path):
    model = torch.jit.load(save_path, map_location='cpu')
    print("export finish.")
    return model

def torch2onnx(model, save_path):
    """
    :param model:
    :param save_path:  XXX/XXX.onnx
    :return:
    """
    model.eval()
    data = torch.rand(1, 3, 224, 224)
    input_names = ["input"]   #ncnn需要
    output_names = ["out"]  #ncnn需要
    torch.onnx._export(model, data, save_path, export_params=True, opset_version=11, input_names=input_names, output_names=output_names)
    print("torch2onnx finish.")

if __name__ == '__main__':
    from models import FPN_ResNet
    from models import ACCL_CB_FPN_ResNet
    import config
    #print_model_param('../../data/PSENet_Resnet_on_ic15/resnet50.pth')
    model = FPN_ResNet(backbone='resnet50', pretrained=False, result_num=6)
    #model = ACCL_CB_FPN_ResNet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.n,
                               # scale=config.scale, checkpoint='../../save/CV/ranger/ranger3/Best_825_r0.767935_p0.854312_f10.808824.pth')

    #load_part_checkpoint('../../save/CV/ranger/ranger3/Best_825_r0.767935_p0.854312_f10.808824.pth', model, device=torch.device('cuda:0'),
                               #part_id_list=[(318, -1)])  # 144, 319

    #print_model_param('../../save/CV/ranger/ranger3/Best_825_r0.767935_p0.854312_f10.808824.pth')
    # load_part_checkpoint('../../save/CV/ranger/ranger3/Best_825_r0.767935_p0.854312_f10.808824.pth', model,
    #                      device=torch.device('cpu'), part_id_list=[(144, 319)]) #[(0, 317), (339, -1)]

    state = torch.load('E:\\PSENet_Resnet_on_ic15\\Best_825_r0.767935_p0.854312_f10.808824.pth', torch.device('cpu'))
    print(state.keys())
    model.load_state_dict(state['state_dict'])
    torch_export(model, 'E:\\PSENet_Resnet_on_ic15\\psenet.pt')
    print('finished')
