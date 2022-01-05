import torch
import shutil
import numpy as np
import config
import os
import cv2
from tqdm import tqdm
from models import FPN_ResNet
from predict_ic15 import Pytorch_model
from cal_recall.script import cal_recall_precison_f1
from utils import draw_bbox
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True


def main(net, model_path, long_size, scale, path, save_path, gpu_id, fast_test):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(path, x) for x in os.listdir(path)]

    model = Pytorch_model(model_path, net=net, scale=scale, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    model_total_time = 0.0
    decode_total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
        #pred, boxes_list, t = model.predict(img_path, long_size=long_size)

        pred, boxes_list, t, model_time, decode_time = model.predict(img_path, long_size=long_size, fast_test=fast_test)
        total_frame += 1
        total_time += t
        model_total_time += model_time
        decode_total_time += decode_time
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))

        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')

    print('fps:{}'.format(total_frame / total_time))
    print('average model time:{}'.format(model_total_time/total_frame))
    print('average decode time:{}'.format(decode_total_time / total_frame))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    long_size = 2200     #2240
    scale = 1   # Best_340_r0.773712_p0.847574_f10.808960.pth
    # model_path = '../save/ic15/vgg_test_distv2_IC15/Best_488_r0.482427_p0.904332_f10.629199.pth'
    model_path = '../.save/IC15/dla2/Best_180_r0.610014_p0.873191_f10.718254.pth'

    data_path = '../data/IC15/test/img'
    gt_path = '../data/IC15/test/gt'   # gt_2pts, gt
    save_path = '../.save/test/ic15/dla2'
    gpu_id = 0
    print('scale:{},model_path:{}'.format(scale,model_path))

    fast_test = True

    from models.craft import CRAFT
    from models.fpn_scnet import FPN_SCNet
    from models.dla_seg import get_dlaseg_net


    # net = FPN_SCNet('scnet50_v1d', 2,  pretrained=False, scale=scale)
    # net = CRAFT(num_out=2, pretrained=False, scale=scale)
    net = get_dlaseg_net(34, heads={'seg_hm': 2})

    save_path = main(net, model_path, long_size, scale, data_path, save_path, gpu_id=gpu_id, fast_test=fast_test)

    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)
    print('scale:', scale)
    print('long_size: ', long_size)

    ############################################
    # import time
    # device = torch.device('cuda:0')  #cuda:0
    # x = torch.randn(1, 3, 512, 512).to(device)
    # start = time.time()
    # y = net(x)
    # print('model prediction time(512*512):', time.time() - start)  # 18->4.5  50->5.8
    #
    # from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary
    #
    # print_model_parm_flops(net, x)
    # print_model_parm_nums(net)
    #show_summary(net, 'E:/summery.xlsx')
    # print(cal_recall_precison_f1('/data2/dataset/ICD151/test/gt', '/data1/zj/tensorflow_PSENet/tmp/'))

# dla2
# 2200  0.295 0.56 0.93  0.95 250
# fps:3.26681
# {'precision': 0.8624733475479744, 'recall': 0.7790081848820414, 'hmean': 0.8186187705540097, 'AP': 0}