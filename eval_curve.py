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
from dist import decode_curve
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
                    sk[k[7:]] = self.net[k]
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
            res_preds, boxes_list = decode_curve(preds[0], scale)
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

    long_size = 1000
    name = 'dla'
    data_type = 'ctw1500'   # ctw1500/total
    model_path = '../.save/ctw1500/'+name+'/Best_93_r0.745763_p0.839633_f10.789919.pth'
    data_path = '../data/ctw1500/test/img'  #../data/totaltext/test/img
    gt_path = '../data/ctw1500/test/gt'   # ../data/totaltext/test/gt
    save_path = '../.save/test/ctw1500/' + name


    long_size = 1350  # 1050
    data_type = 'total'  # ctw1500/total
    name = 'dla_cp'
    # model_path = '../save/Total/distv2_Total_exdata333/Best_164_r0.781843_p0.808123_f10.794766.pth'
    model_path = '../.save/Total/'+name+'/Best_201_r0.747967_p0.840609_f10.791587.pth'
    data_path = '../data/totaltext/test/img'  # ../data/totaltext/test/img
    gt_path = '../data/totaltext/test/gt'  # ../data/totaltext/test/gt
    save_path = '../.save/test/' + name


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
    net = get_dlaseg_net(34, heads={'seg_hm': 2})
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


### Total
# origin2
# 1200  0.285 0.62    0.93, 0.978
# tiouRecall: 0.477 tiouPrecision: 0.63 tiouHmean: 0.543
# {'precision': 0.8304836345872008, 'recall': 0.7678410117434508, 'hmean': 0.7979347570992724}

# origin_adam
# 1200  0.285   0.62   0.93 0.97
# tiouRecall: 0.304 tiouPrecision: 0.478 tiouHmean: 0.372
# {'precision': 0.6879781420765028, 'recall': 0.568654019873532, 'hmean': 0.6226508407517308}

#dla_3
# 1350  0.285   0.56   0.93 0.97
#  2.1467
# tiouRecall: 0.451 tiouPrecision: 0.624 tiouHmean: 0.524
# {'precision': 0.8470185728250245, 'recall': 0.7827461607949413, 'hmean': 0.8136150234741785}

###
### CTW1500
# 1000 0.295   0.56   0.93  0.972 180
# 4.4671
# tiouRecall: 0.452 tiouPrecision: 0.63 tiouHmean: 0.526
# {'precision': 0.840673111349803, 'recall': 0.7653194263363755, 'hmean': 0.8012284593072855}
