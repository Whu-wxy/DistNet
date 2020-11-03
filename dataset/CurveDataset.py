import random
import time
import config
import os
import random
import pathlib
import pyclipper
from torch.utils import data
import torch
import glob
import numpy as np
import cv2
from dataset.augment import PSEDataAugment
from utils.utils import draw_bbox

import config

import Polygon
from Polygon.Utils import pointList

from dataset.data_utils import image_label, image_label_v2, image_label_v3, DataLoaderX

time_sum = 0

class CurveDataset(data.Dataset):
    def __init__(self, data_dir, data_shape: int = 640, dataset_type='ctw1500', transform=None, target_transform=None):
        self.dataset_type = dataset_type
        self.data_list = self.load_data(data_dir)
        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform

        #self.aug = augument()  #20200302增加新augument方式

    def __getitem__(self, index):
        # print(self.image_list[index])
        img_path, text_polys, text_tags = self.data_list[index]
        img, training_mask, distance_map = image_label_v3(img_path, text_polys, text_tags,
                                                                   input_size=self.data_shape,
                                                                   scales = np.array(config.random_scales))

        # global time_sum
        # time_sum += dur

        #img = draw_bbox(img,text_polys)
        #img = self.aug(image=np.array(img))['image']  #20200302增加新augument方式

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            training_mask = self.target_transform(training_mask)

        return img, training_mask, distance_map

    def load_data(self, data_dir: str) -> list:
        data_list = []

        for x in glob.glob(data_dir + '/img/*.jpg', recursive=True):
            d = pathlib.Path(x)
            if self.dataset_type == 'ctw1500':
                label_path = os.path.join(data_dir, 'gt', (str(d.stem) + '.txt'))
            elif self.dataset_type == 'total':
                label_path = os.path.join(data_dir, 'gt', 'poly_gt_' + (str(d.stem) + '.txt'))
                # label_path = os.path.join(data_dir, 'gt', (str(d.stem) + '.txt'))
            else:
                raise Exception('数据集类型必须是ctw1500或total！')
            bboxs, text = self._get_annotation(label_path)
            if len(bboxs) > 0:
                data_list.append((x, bboxs, text))
            else:
                print('there is no suit bbox on {}'.format(label_path))
        return data_list

    def _get_annotation(self, label_path: str) -> tuple:
        boxes = []
        text_tags = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').strip('\ufeff').strip('\xef\xbb\xbf')
                params = line.split(',')
                try:
                    if self.dataset_type == 'ctw1500':
                        text_tags.append(False)
                        xmin, ymin, w, h = list(map(float, params[:4]))
                        box = []
                        x = 0
                        y = 0
                        for i, val in enumerate(params[4:]):
                            if i % 2 == 0:
                                x = xmin + int(val)
                            elif i % 2 == 1:
                                y = ymin + int(val)
                                box.append([x, y])
                        boxes.append(box)
                    elif self.dataset_type == 'total':
                        box = []
                        x = 0
                        y = 0
                        if not params[-1].isdigit():
                            if line.endswith('#'):
                                text_tags.append(True)
                            else:
                                text_tags.append(False)
                            params.pop(-1)
                        else:
                            text_tags.append(False)
                        for i, val in enumerate(params):
                            if i % 2 == 0:
                                x = int(val)
                            elif i % 2 == 1:
                                y = int(val)
                                box.append([x, y])
                        boxes.append(box)

                except:
                    print('load label failed on {}'.format(label_path))

        if self.dataset_type == 'total':   # padding
            pt_count = 0
            for box in boxes:
                if len(box) > pt_count:
                    pt_count = len(box)
            for i, box in enumerate(boxes):
                if len(box) < pt_count:
                    box2 = box
                    for j in range(len(box), pt_count):
                        box2.append(box[-1])
                    boxes[i] = box2

        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img



if __name__ == '__main__':
    import torch
    import config
    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    from collections import Counter

    # print('1')
    # for x in glob.glob('F:\imgs\ps\*.jpg|*.png)', recursive=False):
    #     print(x)
    # input()

#F:\\imgs\\psenet_vis2s     F:\zzxs\dl-data\ICDAR\ICDAR2015\\train
    #F:\zzxs\dl-data\ICDAR\ICDAR2015\sample_IC15\\train
    #F:\zzxs\Experiments\dl-data\CTW
    #F:\zzxs\Experiments\dl-data\CTW\ctw1500\\train
    # F:\zzxs\Experiments\dl-data\TotalText\\train
    train_data = CurveDataset('../../data/ReCTS/ReCTS_OUR/train', data_shape=config.data_shape,
                            dataset_type='total', transform=transforms.ToTensor())
    train_loader = DataLoaderX(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    empty_count = 0
    max_values = []

    for i, (img, mask, distance_map) in enumerate(train_loader):
        pbar.update(1)
        # print(img.shape)  # BCWH
        # print(mask.shape)       #BWH
        #
        # print(distance_map.shape)  #BWH
        #print(dist_maps.shape)
        # input()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dist_map", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow('img', img.squeeze(0).numpy().transpose((1, 2, 0)))
        #cv2.imshow('mask', mask.numpy().transpose((1, 2, 0))*255)
        distance_map = distance_map.numpy().transpose((1, 2, 0))
        cv2.imshow('dist_map', distance_map)
        cv2.waitKey()
        cv2.destroyAllWindows()

        center = np.where(distance_map>=0.6, 255, 0).astype(np.uint8)
        center2 = np.where(distance_map >= 0.8, 255, 0).astype(np.uint8)
        #cv2.imwrite('F:\zzxs\Experiments\dl-data\CTW\\' + str(i) + 'dist.jpg', distance_map.numpy().transpose((1, 2, 0))*255)

        distance_map = distance_map*255
        distance_map = distance_map.astype(np.uint8)
        distance_map = cv2.cvtColor(distance_map, cv2.COLOR_GRAY2BGR)

        contours, hierarchy = cv2.findContours(center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(distance_map, contours, -1, (0, 0, 255), 1)
        contours, hierarchy = cv2.findContours(center2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(distance_map, contours, -1, (0, 255, 0), 1)

        cv2.imwrite('F:\zzxs\Experiments\dl-data\CTW\\exp' + str(i) + 'dist.jpg', distance_map)


    pbar.close()
    print('all time:', time_sum)
    print('count:', len(train_loader))
    print('ave time:', time_sum/len(train_loader))

