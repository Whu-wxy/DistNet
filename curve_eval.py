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

from cal_recall.curve_script import curve_cal_recall_precison_f1
from utils import draw_bbox
from dist import decode_curve as dist_decode_curve

torch.backends.cudnn.benchmark = True


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
            start = time.time()
            preds = self.net(tensor)
            # print(preds)
            # return None, None, None

            #preds, boxes_list = pse_decode(preds[0], self.scale)
            preds, boxes_list = dist_decode_curve(preds[0], self.scale)
            scale = (preds.shape[1] / w, preds.shape[0] / h)

            if len(boxes_list):
                boxes_list = boxes_list / scale
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.time() - start
        return preds, boxes_list, t  #, logit


def main(net, model_path, long_size, scale, path, save_path, gpu_id):
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
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, img_name + '.txt')
        pred, boxes_list, t = model.predict(img_path, long_size=long_size)
        total_frame += 1
        total_time += t
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, boxes_list.shape[-1]*boxes_list.shape[-2]), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


if __name__ == '__main__':
    from models.GFF_FPN import GFF_FPN
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    long_size = 1280
    scale = 4
    data_type = 'ctw1500'   # ctw1500/total
    model_path = '../save/dist_vgg_region_fixwse/Best_536_r0.743861_p0.834684_f10.786660.pth'

    #../ save / dist_gff / Best_624_r0.636976_p0.580518_f10.607438.pth

    data_path = '../IC15/test/img'
    gt_path = '../IC15/test/gt'   # gt_2pts, gt
    save_path = '../test_result'
    gpu_id = 0
    print('scale:{},model_path:{}'.format(scale,model_path))

    #net = GFF_FPN(backbone=backbone, pretrained=False, result_num=config.n)
    from models.craft import CRAFT

    net = CRAFT(num_out=2, pretrained=False)

    save_path = main(net, model_path, long_size, scale, data_path, save_path, gpu_id=gpu_id)

    # ctw1500/total
    result = curve_cal_recall_precison_f1(type=data_type,gt_path=gt_path, result_path=save_path)
    print(result)
    print('scale:', scale)
    print('long_size: ', long_size)

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

