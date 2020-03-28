import torch
import shutil
import numpy as np
import config
import os
import cv2
from tqdm import tqdm
from models import FPN_ResNet
from predict import Pytorch_model
from cal_recall.script import cal_recall_precison_f1
from cal_recall.script_13 import cal_recall_precison_f1_13
from cal_recall.script_deteval import cal_recall_precison_f1_deteval
from utils import draw_bbox

torch.backends.cudnn.benchmark = True


def main(net, model_path, backbone, long_size, scale, path, save_path, gpu_id):
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
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
        pred, boxes_list, t = model.predict(img_path, long_size=long_size)
        total_frame += 1
        total_time += t
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        if config.save_4_pt_box:
            np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
        else:
            np.savetxt(save_name, boxes_list.reshape(-1, 4), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


if __name__ == '__main__':
    from models.GFF_FPN import GFF_FPN
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    backbone = 'resnet50'  #res2net50_26w_6s   res2net_dla60
    long_size = 2200     #2240
    scale = 1
    eval_script = 'iou'
    model_path = '../save/dist_biregion/Best_460_r0.654309_p0.622253_f10.637878.pth'

    #../ save / dist_gff / Best_624_r0.636976_p0.580518_f10.607438.pth

    data_path = '../IC15/test/img'
    gt_path = '../IC15/test/gt'   # gt_2pts, gt
    save_path = '../save/test_result2'
    gpu_id = 0
    print('backbone:{},scale:{},model_path:{}'.format(backbone,scale,model_path))

    #net = GFF_FPN(backbone=backbone, pretrained=False, result_num=config.n)
    net = FPN_ResNet(backbone=backbone, pretrained=False, result_num=config.n)

    save_path = main(net, model_path, backbone, long_size, scale, data_path, save_path, gpu_id=gpu_id)

    if eval_script == 'iou':
        result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
        print('iou eval.')
    elif eval_script == 'deteval':
        result = cal_recall_precison_f1_deteval(gt_path=gt_path, result_path=save_path)
        print('deteval eval.')
    elif eval_script == '2013':
        result = cal_recall_precison_f1_13(gt_path=gt_path, result_path=save_path)
        print('2013 eval.')
    print(result)
    print('scale:', scale)
    print('long_size: ', long_size)
    print('min_threld: ', config.min_threld)
    print('max_threld: ', config.max_threld)

    ############################################
    import time
    device = torch.device('cuda:0')  #cuda:0
    x = torch.randn(1, 3, 512, 512).to(device)
    start = time.time()
    y = net(x)
    print('model prediction time(512*512):', time.time() - start)  # 18->4.5  50->5.8

    from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

    print_model_parm_flops(net, x)
    print_model_parm_nums(net)
    #show_summary(net, 'E:/summery.xlsx')
    # print(cal_recall_precison_f1('/data2/dataset/ICD151/test/gt', '/data1/zj/tensorflow_PSENet/tmp/'))

