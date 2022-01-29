import torch
from torchvision import transforms
import shutil
import numpy as np
import config
import os
import cv2
from tqdm import tqdm
from models import FPN_ResNet
import utils
import timeit

from cal_recall.curve_script import curve_cal_recall_precison_f1
from utils import draw_bbox
from dist import decode_curve, fast_decode_curve
import matplotlib.pyplot as plt

from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

torch.backends.cudnn.benchmark = True

def write_result_as_txt(save_path, bboxes):
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d"%values[0]
        for v_id in range(1, len(values)):
            line += ", %d"%values[v_id]
        line += '\n'
        lines.append(line)
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)


class Pytorch_model_curve:
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
                    if 'base.fc' in k:
                        print(k)
                        continue

                    sk[k] = self.net[k]
                    # sk[k[7:]] = self.net[k]
                net.load_state_dict(sk)
            except:
                net.load_state_dict(self.net)
            self.net = net
            print('load models')
        self.net.eval()

    def predict(self, img: str, long_size: int = 2240, fast_test=True):
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

        # pad
        if config.dla_model:
            h, w = img.shape[:2]
            h_pad, w_pad = 0, 0
            pad_to_scale = 32
            if h % pad_to_scale != 0:
                h_pad = (h // pad_to_scale + 1) * pad_to_scale - h
            if w % pad_to_scale != 0:
                w_pad = (w // pad_to_scale + 1) * pad_to_scale - w
            img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)))
            h, w, _ = img.shape

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

            decode_time = timeit.default_timer()
            res_preds, boxes_list = decode_curve(preds[0], scale)  # fast_
            decode_time = (timeit.default_timer() - decode_time)

            if not fast_test:
                decode_time = timeit.default_timer()
                for i in range(30):  # same as DBNet: https://github.com/MhLiao/DB/blob/master/eval.py
                    preds_temp, boxes_list = decode_curve(preds[0], scale)
                decode_time = (timeit.default_timer() - decode_time) / 30.0

            t = model_time + decode_time

        return preds, boxes_list, t, model_time, decode_time  #, logit


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

    model = Pytorch_model_curve(model_path, net=net, scale=scale, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    model_total_time = 0.0
    decode_total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, img_name + '.txt')
        if os.path.exists(save_name):
            continue

        pred, boxes_list, t, model_time, decode_time = model.predict(img_path, long_size=long_size, fast_test=fast_test)
        total_frame += 1
        total_time += t
        model_total_time += model_time
        decode_total_time += decode_time

        text_box = None
        if isinstance(img_path, str):
            text_box = cv2.imread(img_path)

        for bbox in boxes_list:
            cv2.drawContours(text_box, [bbox.reshape(bbox.shape[0] // 2, 2)], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), text_box)

        write_result_as_txt(save_name, boxes_list)

    print('fps:{}'.format(total_frame / total_time))
    print('average model time:{}'.format(model_total_time / total_frame))
    print('average decode time:{}'.format(decode_total_time / total_frame))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    scale = 1

    long_size = 1150
    name = 'dla_CNN'
    # dla   Best_93_r0.745763_p0.839633_f10.789919.pth
    # dla_CNN  Best_159_r0.721317_p0.853452_f10.781841.pth
    data_type = 'ctw1500'   # ctw1500/total
    model_path = '../.save/ctw1500/'+name+'/Best_159_r0.721317_p0.853452_f10.781841.pth'
    data_path = '../data/ctw1500/test/img'  #../data/totaltext/test/img
    gt_path = '../data/ctw1500/test/gt'   # ../data/totaltext/test/gt
    save_path = '../.save/test/ctw1500/' + name

# dla_down_ratio4_rigid   Best_180_r0.763324_p0.833333_f10.796794.pth
    # dla34_3  Best_162_r0.762421_p0.838549_f10.798675.pth
    # dla_head256_fsm  Best_171_r0.757001_p0.848608_f10.800191.pth
    # dla_head256_woDCN Best_174_r0.754291_p0.818627_f10.785143.pth
    # long_size = 1350  # 1050
    # data_type = 'total'  # ctw1500/total
    # name = 'dla_head256_woDCN'
    # # # model_path = '../save/Total/distv2_Total_exdata333/Best_164_r0.781843_p0.808123_f10.794766.pth'
    # # # Best_162_r0.762421_p0.838549_f10.798675.pth
    # model_path = '../.save/Total/'+name+'/Best_174_r0.754291_p0.818627_f10.785143.pth'
    # data_path = '../data/totaltext/test/img'  # ../data/totaltext/test/img
    # gt_path = '../data/totaltext/test/gt'  # ../data/totaltext/test/gt
    # save_path = '../.save/test/total/' + name #+ '_temp'


    gpu_id = 0
    print('scale:{},model_path:{}'.format(scale,model_path))

    fast_test = True

    from models.craft import CRAFT
    from models.fapn_resnet import FaPN_ResNet
    from models.fapn_vgg16 import FaPN_VGG16_bn
    from models.dla_seg import get_dlaseg_net

    # net = CRAFT(num_out=2, pretrained=False)
    # net = FaPN_ResNet("resnet50", 2, 1, True)
    # net = FaPN_VGG16_bn(num_out=2, pretrained=False)
    net = get_dlaseg_net(34, heads={'seg_hm': 2}, down_ratio=4, head_conv=256, bFSM=False)
    save_path = main(net, model_path, long_size,
                     scale, data_path, save_path, gpu_id=gpu_id, fast_test=fast_test)

    print('save path:', save_path)

    # ctw1500/total
    result = curve_cal_recall_precison_f1(type=data_type, gt_path=gt_path, result_path=save_path)
    print(result)
    print('scale:', scale)
    print('long_size: ', long_size)
    #
    # ############################################
    # import time
    # device = torch.device('cuda:0')  #cuda:0
    # x = torch.randn(1, 3, 512, 512).to(device)
    # start = time.time()
    # y = net(x)
    # print('model prediction time(512*512):', time.time() - start)  # 18->4.5  50->5.8
    #
