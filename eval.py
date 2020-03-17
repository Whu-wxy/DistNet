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
        _, boxes_list, t = model.predict(img_path, long_size=long_size)
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    backbone = 'resnet50'  #res2net50_26w_6s   res2net_dla60
    long_size = 2240     #2240
    scale = 1
    model_path = '../Best_558_r0.662494_p0.583793_f10.620659.pth'
    #../data/PSENet_Resnet_on_ic15/resnet152.pth  #Best_638_r0.584497_p0.829235_f10.685682.pth
    #res2net50_26w_6s_2.1/train_PSENet_524_loss0.347171.pth
    #../save/CV/res2net_dla2.0/Best_638_r0.584497_p0.829235_f10.685682.pth
    data_path = '../IC15/test/img'
    gt_path = '../IC15/test/gt'   # gt_2pts, gt
    save_path = '../save/test_result2'
    gpu_id = 0
    print('backbone:{},scale:{},model_path:{}'.format(backbone,scale,model_path))

    net = FPN_ResNet(backbone=backbone, pretrained=False, result_num=config.n)
    # net = ACCL_CB_FPN_ResNet(backbone=backbone, pretrained=False, result_num=config.n,
    #                            scale=config.scale,
    #                            checkpoint='')
    # net = dla60up(classes=config.n, scale=config.scale, pretrained_base=False)

    save_path = main(net, model_path, backbone, long_size, scale, data_path, save_path, gpu_id=gpu_id)

    if config.eval_script == 'iou':
        result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
        print('iou eval.')
    elif config.eval_script == 'deteval':
        result = cal_recall_precison_f1_deteval(gt_path=gt_path, result_path=save_path)
    elif config.eval_script == '2013':
        result = cal_recall_precison_f1_13(gt_path=gt_path, result_path=save_path)
    print(result)
    print('decode_threld: ', config.decode_threld)

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

