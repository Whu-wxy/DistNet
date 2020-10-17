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
from tqdm import tqdm
import config

import Polygon
from Polygon.Utils import pointList

from dataset.data_utils import image_label, image_label_v2, image_label_v3, DataLoaderX

time_sum = 0

class SynthTextDataset(data.Dataset):
    def __init__(self, rootpath, data_shape: int = 640, transform=None, target_transform=None):
        number = 0
        self.root_path = rootpath
        with open(os.path.join(rootpath, "train_list.txt"), "r") as f:
            # 获得训练数据的总行数
            for _ in tqdm(f, desc="load training dataset"):
                number += 1
        self.number = number
        self.fopen = open(os.path.join(rootpath, "train_list.txt"), 'r')

        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        line = self.fopen.__next__()
        #8/ballet_106_0.jpg, ballet_106_0.txt
        line = line.split(',')
        img_path = line[0].strip()
        gt_path = line[1].strip()

        img_path = os.path.join(self.root_path, "img", img_path)
        gt_path = os.path.join(self.root_path, "gt", gt_path)

        text_polys, text_tags = self._get_annotation(gt_path)
        try:
            img, training_mask, distance_map = image_label_v3(img_path, text_polys, text_tags,
                                                                   input_size=self.data_shape,
                                                                   scales = np.array(config.random_scales))
        except:
            print('error: ', img_path)


        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            training_mask = self.target_transform(training_mask)

        return img, training_mask, distance_map

    def _get_annotation(self, label_path: str) -> tuple:
        boxes = []
        text_tags = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                if ',' in line:
                    params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                else:
                    params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(' ')
                try:
                    if len(params) >= 8:
                        label = params[-1]
                        if label == '*' or label == '###':   #在loss中用mask去掉
                            text_tags.append(True)  # True
                        else:
                            text_tags.append(False)  # False
                        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                    boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                except:
                    print('load label failed on {}'.format(label_path))
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return self.number



if __name__ == '__main__':
    import torch
    import config
    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    from collections import Counter

    train_data = SynthTextDataset('F:\\zzxs\\Experiments\\dl-data\\SynthText\\SynthText800k\\detection\\SynthText', data_shape=config.data_shape,
                             transform=transforms.ToTensor())
    train_loader = DataLoaderX(dataset=train_data, batch_size=1, shuffle=False, pin_memory=True)

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

        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("dist_map", cv2.WINDOW_NORMAL)
        # #cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img.squeeze(0).numpy().transpose((1, 2, 0)))
        # #cv2.imshow('mask', mask.numpy().transpose((1, 2, 0))*255)
        # cv2.imshow('dist_map', distance_map.numpy().transpose((1, 2, 0)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # cv2.imwrite('F:\zzxs\Experiments\dl-data\CTW\\' + str(i) + 'dist.jpg', distance_map.numpy().transpose((1, 2, 0))*255)

    pbar.close()
    print('all time:', time_sum)
    print('count:', len(train_loader))
    print('ave time:', time_sum/len(train_loader))