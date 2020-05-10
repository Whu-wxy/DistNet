# -*- coding: utf-8 -*-
# @Time    : 1/4/19 11:14 AM
# @Author  : zhoujun
import torch
from torchvision import transforms
import os
import cv2
import timeit
import numpy as np
import utils
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

from dist import decode as dist_decode
from dist import decode_curve as dist_decode_curve

class Pytorch_model:
    def __init__(self, model_path, net, scale, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.scale = scale
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")

        #self.net = net.to(self.device)
        self.net = torch.load(model_path, map_location=self.device)['state_dict']
        #self.net = torch.jit.load(model_path, map_location=self.device)

        print('device:', self.device)

        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            net.scale = scale
            try:
                sk = {}
                for k in self.net:
                    sk[k[7:]] = self.net[k]
                net.load_state_dict(sk)
            except:
                net.load_state_dict(self.net)
            self.net = net
            print('load models')
        self.net.eval()


    def predict(self, img: str, long_size: int = 2240):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img), 'file is not exists'
        if img.endswith('jpg'):
            in_file = open(img, 'rb')
            img = jpeg.decode(in_file.read())
            in_file.close()
            # im = jpeg.JPEG(im_fn).decode()
        else:
            img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if long_size != None:
            scale = long_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            model_time = timeit.default_timer()
            preds = self.net(tensor)
            model_time = (timeit.default_timer() - model_time)

            res_preds, boxes_list, scores_list = dist_decode(preds[0], self.scale)

            decode_time = timeit.default_timer()
            for i in range(50):    # same as DBNet: https://github.com/MhLiao/DB/blob/master/eval.py
                preds_temp, boxes_list, scores_list = dist_decode(preds[0], self.scale)
            decode_time = (timeit.default_timer() - decode_time) / 50.0

            t = model_time + decode_time

            scale = (res_preds.shape[1] / w, res_preds.shape[0] / h)

            if len(boxes_list):
                boxes_list = boxes_list / scale

        return res_preds, boxes_list, t, model_time, decode_time  #, logit


def _get_annotation(label_path):
    boxes = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                label = params[8]
                if label == '*' or label == '###':
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            except:
                print('load label failed on {}'.format(label_path))
    return np.array(boxes, dtype=np.float32)


if __name__ == '__main__':
    import config
    from models import FPN_ResNet
    import matplotlib.pyplot as plt
    from utils.utils import show_img, draw_bbox


    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')

    model_path = '../Best_340_r0.773712_p0.847574_f10.808960.pth'   #psenet.pt

    img_id = 10
    img_path = '../data/IC15/test/img/img_{}.jpg'.format(img_id)
    label_path = '../data/IC15/test/gt/gt_img_{}.txt'.format(img_id)
  #  label = _get_annotation(label_path)

    # 初始化网络
    from models.craft import CRAFT

    net = CRAFT(num_out=2, pretrained=False)
    model = Pytorch_model(model_path, net=net, scale=1, gpu_id=0)
    # for i in range(100):
    #     models.predict(img_path)
    preds, boxes_list,t, model_time, decode_time = model.predict(img_path, 1800)
    #print(boxes_list)
    # show_img(preds)

    cv2.imwrite('../img_pred14.jpg', logit*255)

    center = np.where(logit>=config.max_threld, 255, 0)
    cv2.imwrite('../img_pred_center14.jpg', center)

    region = np.where(logit >= config.min_threld, 255, 0)
    cv2.imwrite('../img_pred_region14.jpg', region)

    img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
    cv2.imwrite('../img_label14.jpg', img)
#    img = draw_bbox(img, label,color=(0,0,255))
#     show_img(img, color=True)
#
#     plt.show()
