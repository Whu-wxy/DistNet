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

class IC17Dataset(data.Dataset):
    def __init__(self, train_dir, validation_dir=None, data_shape: int = 640, transform=None, target_transform=None):
        self.data_list = self.load_data(train_dir)
        if validation_dir != None:
            self.data_list.extend(self.load_data(validation_dir))
        print('count:', len(self.data_list))
        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform

        #self.aug = augument()  #20200302增加新augument方式

    def __getitem__(self, index):
        img_path, text_polys, text_tags = self.data_list[index]
        try:
            img, training_mask, distance_map = image_label_v3(img_path, text_polys, text_tags,
                                                                   input_size=self.data_shape,
                                                                   scales = np.array(config.random_scales))
        except:
            print('error: ', img_path)


        #img = draw_bbox(img,text_polys)
        #img = self.aug(image=np.array(img))['image']  #20200302增加新augument方式

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            training_mask = self.target_transform(training_mask)

        return img, training_mask, distance_map

    def load_data(self, data_dir: str) -> list:
        data_list = []
        img_list = os.listdir(data_dir + '/img')   # jpg and png
        for x in img_list:
        #for x in glob.glob(data_dir + '/img/*.jpg', recursive=True):
            d = pathlib.Path(x)
            label_path = os.path.join(data_dir, 'gt', ('gt_' + str(d.stem) + '.txt'))
            # label_path = os.path.join(data_dir, 'gt', (str(d.stem) + '.txt'))

            bboxs, text = self._get_annotation(label_path)
            if len(bboxs) > 0:
                x_path = os.path.join(data_dir, 'img', x)
                data_list.append((x_path, bboxs, text))
            else:
                print('there is no suit bbox on {}'.format(label_path))
        return data_list

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
                        # text_tags.append(False)
                        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                    boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                except:
                    print('load label failed on {}'.format(label_path))
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)



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
    # F:\zzxs\Experiments\dl-data\TotalText\\train    #validation
    train_data = IC17Dataset('../../data/IC17/train', '../../data/IC17/validation', data_shape=config.data_shape,
                             transform=transforms.ToTensor())
    train_loader = DataLoaderX(dataset=train_data, batch_size=6, shuffle=False, num_workers=10)

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