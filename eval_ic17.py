import torch
import shutil
import numpy as np
import config
import os
import cv2
from tqdm import tqdm
from models import FPN_ResNet
from torchvision import transforms
from predict_ic15 import Pytorch_model
from cal_recall.script17 import cal_recall_precison_f1_17
from utils import draw_bbox
from dist import decode as dist_decode
import timeit
import Polygon

from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

torch.backends.cudnn.benchmark = True

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

def write_result_as_txt(save_path, bboxes, scores):
    lines = []

    covered_list = []
    uncovered_list = []
    for (i, box) in enumerate(bboxes):
        if i in covered_list:
            continue

        p = Polygon.Polygon(box)
        for j, tempBox in enumerate(bboxes):
            if j == i:
                continue
            if j in covered_list:
                continue
            p2 = Polygon.Polygon(tempBox)
            if p.covers(p2):
                covered_list.append(j)

    for b_idx, bbox in enumerate(bboxes):
        line = ''
        if b_idx in covered_list:
            continue

        for box in bbox:
            line += "%d, %d, "%(int(box[0]), int(box[1]))

        score = 1
        if b_idx>len(scores)-1:
            score = 1
        else:
            score = scores[b_idx]
        line += str(score)
        line += '\n'
        lines.append(line)
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)


class Pytorch_model_17:
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

    def predict(self, img_path: str, long_size: int = 2240, fast_test=True):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        if img_path.endswith('jpg'):
            in_file = open(img_path, 'rb')
            img = jpeg.decode(in_file.read())
            in_file.close()
            # im = jpeg.JPEG(im_fn).decode()
        else:
            img = cv2.imread(img_path)

        try:
            #img = np.asarray(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print('error: ', img_path)
        h, w = img.shape[:2]

        if long_size != None:
            scale = long_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        #img = cv2.resize(img, None, fx=2, fy=2)

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
            res_preds, boxes_list, scores_list = dist_decode(preds[0], scale)
            decode_time = (timeit.default_timer() - decode_time)

            if not fast_test:
                decode_time = timeit.default_timer()
                for i in range(50):  # same as DBNet: https://github.com/MhLiao/DB/blob/master/eval.py
                    preds_temp, boxes_list, scores_list = dist_decode(preds[0], scale)
                decode_time = (timeit.default_timer() - decode_time) / 50.0

            t = model_time + decode_time

            scale = (res_preds.shape[1] / w, res_preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
        return res_preds, boxes_list, t, scores_list, model_time, decode_time  #, logit


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

    model = Pytorch_model_17(model_path, net=net, scale=scale, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    model_total_time = 0.0
    decode_total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        lab_name = 'res_img_' + img_name.split('_')[-1]
        save_name = os.path.join(save_txt_folder, lab_name + '.txt')
        # print(lab_name)
        # input()
        if os.path.exists(save_name):
            # print('exist')
            continue

        pred, boxes_list, t, scores_list, model_time, decode_time = model.predict(img_path, long_size=long_size, fast_test=fast_test)
        total_frame += 1
        total_time += t
        model_total_time += model_time
        decode_total_time += decode_time

        # text_box = None
        # if isinstance(img_path, str):
        #     text_box = cv2.imread(img_path)
        # text_box = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        # cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), text_box)

        write_result_as_txt(save_name, boxes_list, scores_list)

    print('fps:{}'.format(total_frame / total_time))
    print('average model time:{}'.format(model_total_time / total_frame))
    print('average decode time:{}'.format(decode_total_time / total_frame))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    long_size = 2800     #2240
    scale = 1    #DistNet_IC17_130_loss1.029557.pth
    model_path = '../save/dist_IC17_3/DistNet_IC17_150_loss1.043292.pth'   #DistNet_IC17_97_loss1.057110.pth

    data_path = '../data/IC17/test/img'
    gt_path = '../data/IC17/test/gt'   # gt_2pts, gt
    save_path = '../test_result150'
    gpu_id = 0
    print('scale:{},model_path:{}'.format(scale,model_path))

    fast_test=True

    from models.craft import CRAFT

    net = CRAFT(num_out=2, pretrained=False, scale=scale)

    save_path = main(net, model_path, long_size, scale, data_path, save_path, gpu_id=gpu_id, fast_test=fast_test)

    # result = cal_recall_precison_f1_17(gt_path=gt_path, result_path=save_path)
    # print(result)
    # print('scale:', scale)
    # print('long_size: ', long_size)
    #
    # ############################################
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

