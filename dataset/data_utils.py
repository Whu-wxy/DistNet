# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun

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

# from albumentations import (
#     Compose, RGBShift, RandomBrightness, RandomContrast,
#     HueSaturationValue, ChannelShuffle, CLAHE,
#     RandomContrast, Blur, ToGray, JpegCompression,
#     CoarseDropout, RandomRotate90
# )

data_aug = PSEDataAugment()

dur = 0

from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

# def augument():
#     augm = Compose([
#         RGBShift(),
#         RandomBrightness(),
#         RandomContrast(),
#         HueSaturationValue(p=0.2),
#         ChannelShuffle(),
#         CLAHE(),
#         Blur(),
#         ToGray(),
#         CoarseDropout()
#     ],
#     p=0.5)
#     return augm

def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)


def generate_rbox(im_size, text_polys, text_tags, training_mask):
    """
    生成mask图，白色部分是文本，黑色是背景
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):
        poly = poly.astype(np.int)

        cv2.fillPoly(score_map, [poly], 1)
        if tag:
            cv2.fillPoly(training_mask, [poly], 0)

    return score_map, training_mask


def augmentation(im: object, text_polys: object, scales: object, degrees: object, input_size: object) -> object:
    #the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    if 'resize' in config.augment_list:
        im, text_polys = data_aug.random_scale(im, text_polys, scales)
    #the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5 and 'flip' in config.augment_list:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5 and 'rotate' in config.augment_list:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    if random.random() < 0.1 and 'rotate90' in config.augment_list:
        im, text_polys = data_aug.random_rotate90_img_bbox(im, text_polys)

    return im, text_polys


def image_label(im_fn: str, text_polys: np.ndarray, text_tags: list, input_size: int,
                degrees: int = 10, scales: np.ndarray = np.array([0.5, 1, 2.0, 3.0])) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''

    im = cv2.imread(im_fn)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # 检查越界
    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys = augmentation(im, text_polys, scales, degrees, input_size)

    h, w, _ = im.shape
    short_edge = min(h, w)
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    h, w, _ = im.shape

    #####################################
    #start = time.time()

    overlap_map = np.zeros((h, w), dtype=np.uint8)
    for (i, text_poly) in enumerate(text_polys):
        p = Polygon.Polygon(text_poly.astype(np.int))
        for tempPoly in text_polys[i+1:]:
            p2 = Polygon.Polygon(tempPoly)
            if p.overlaps(p2):
                pts = pointList(p&p2)
                pts2 = []
                for pt in pts:
                    pts2.append([int(pt[0]), int(pt[1])])
                overlap_poly = np.array([pts2])
                cv2.fillPoly(overlap_map, overlap_poly, 1)

    score_maps_line = np.zeros((h, w), dtype=np.uint8)
    for text_poly in text_polys:
        pts = []
        for pt in text_poly:
            pts.append([int(pt[0]), int(pt[1])])
        cv2.polylines(score_maps_line, np.array([pts]), True, 1, 1, lineType=cv2.LINE_8)

    #mid = time.time()
    #####################################

    # normal images
    if config.img_norm:
        im = im.astype(np.float32)
        im /= 255.0
        im -= np.array((0.485, 0.456, 0.406))
        im /= np.array((0.229, 0.224, 0.225))

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    score_map, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask)
    score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)

    ##############################
    #mid2 = time.time() - mid
    distance_map = get_distance_map(score_maps, overlap_map, score_maps_line)
    #global dur
    #dur += time.time() - start - mid2

    ##############################
    imgs = data_aug.random_crop_author([im,training_mask, np.expand_dims(distance_map, 2)], (input_size, input_size))
    #score_maps = np.squeeze(imgs[1], 2)


    #return im, training_mask, distance_map
    return imgs[0], imgs[1], np.squeeze(imgs[2], 2)   #im,training_mask#


def get_distance_map(label, overlap_map, score_maps_line):
    masklarge = label[0].astype(np.uint8)  # .transpose((1, 2, 0))
    overlap_map = overlap_map.astype(np.uint8)  # .transpose((1, 2, 0))
    score_maps_line = score_maps_line.astype(np.uint8)
    interMask = masklarge - score_maps_line - overlap_map

    #距离图
    distance_inter_map = cv2.distanceTransform(interMask, distanceType=cv2.DIST_L2, maskSize=5)

    #找到所有子区域
    connect_num, connect_img = cv2.connectedComponents(distance_inter_map.astype(np.uint8), connectivity=4)

    #子区域内部0.4-1
    for lab in range(connect_num):
        if lab == 0:
            continue
        lab_img_i = np.where(connect_img == lab, 1, 0)
        cv2.normalize(distance_inter_map, distance_inter_map, 0.4, 1, cv2.NORM_MINMAX, mask=lab_img_i.astype(np.uint8))

    #相交区域
    score_maps_line2 = np.where(overlap_map == 1, 0.3, 0)
    distance_map = (1 - overlap_map) * distance_inter_map + score_maps_line2

    #边界
    score_maps_line2 = np.where(score_maps_line == 1, 0.3, 0)
    distance_map = (1 - score_maps_line) * distance_map + score_maps_line2

    #
    # np.savetxt('F:\\distance_map.csv', distance_map, delimiter=',', fmt='%F')
    # input()
    return distance_map



def image_label_v2(im_fn: str, text_polys: np.ndarray, text_tags: list, input_size: int,
                degrees: int = 10, scales: np.ndarray = np.array([0.5, 1, 2.0, 3.0])) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''

    im = cv2.imread(im_fn)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # 检查越界
    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys = augmentation(im, text_polys, scales, degrees, input_size)

    h, w, _ = im.shape
    short_edge = min(h, w)
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    h, w, _ = im.shape

    # normal images
    if config.img_norm:
        im = im.astype(np.float32)
        im /= 255.0
        im -= np.array((0.485, 0.456, 0.406))
        im /= np.array((0.229, 0.224, 0.225))

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):
        poly = poly.astype(np.int)
        if tag:
            cv2.fillPoly(training_mask, [poly], 0)

    #####################################
    # start = time.time()
    distance_map = get_distance_map_v2(text_polys, h, w)
    # global dur
    # dur += time.time() - start

    ##############################
    imgs = data_aug.random_crop_author([im,training_mask, np.expand_dims(distance_map, 2)], (input_size, input_size))

    #return im, training_mask, distance_map
    return imgs[0], imgs[1], np.squeeze(imgs[2], 2)   #, time.time() - start   #im,training_mask#


def get_distance_map_v2(text_polys, h, w):
    dist_map = np.zeros((h, w), dtype=np.float)

    for (i, text_poly) in enumerate(text_polys, start=1):
        temp_map = np.zeros((h, w), dtype=np.float)
        pts = []
        for pt in text_poly:
            pts.append([int(pt[0]), int(pt[1])])
        cv2.fillPoly(temp_map, np.array([pts]), i)

        Intersection = np.where((dist_map > 0.01) & (temp_map > 0.01), 1, 0)
        Inter_count = np.sum(Intersection)
        if Inter_count == 0:
            dist_map[temp_map==i] = i
            continue

        for j in range(1, i):
            inter = np.where((dist_map == j) & (temp_map > 0.01), 1, 0)
            inter_sum = np.sum(inter)
            if inter_sum == 0:       # 找出已绘制box中和当前box有交集的
                continue

            inter_region = np.where(dist_map==j, 1, 0)
            inter_region_sum = np.sum(inter_region)
            temp_map_sum = np.sum(temp_map)
            rate_temp = float(inter_sum) / temp_map_sum
            rate_inter_region = float(inter_sum) / inter_region_sum
            if rate_temp > rate_inter_region:
                dist_map[temp_map==i] = i
            else:
                dist_map[(temp_map==i)&(inter!=1)] = i
            if inter_sum == Inter_count:  # 当前box只与这个box相交
                break

    for (i, text_poly) in enumerate(text_polys, start=1):
        text_i = np.where(dist_map == i, 1, 0)
        text_i = text_i.astype(np.uint8)
        # 距离图
        distance_map_i = cv2.distanceTransform(text_i, distanceType=cv2.DIST_L2, maskSize=5)
        cv2.normalize(distance_map_i, distance_map_i, 0.3, 1, cv2.NORM_MINMAX, mask=text_i.astype(np.uint8))
        dist_map = dist_map * (1 - text_i) + distance_map_i
    #
    # np.savetxt('F:\\distance_map.csv', distance_map, delimiter=',', fmt='%F')
    # input()
    return dist_map


#import jpeg4py as jpeg
import timeit

def image_label_v3(im_fn: str, text_polys: np.ndarray, text_tags: list, input_size: int,
                degrees: int = 10, scales: np.ndarray = np.array([0.5, 1, 2.0, 3.0])) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''

    #start = time.time()
    if im_fn.endswith('jpg'):
        try:
            in_file = open(im_fn, 'rb')
            im = jpeg.decode(in_file.read())
            in_file.close()
            #im = jpeg.JPEG(im_fn).decode()
            #im = cv2.imread(im_fn)
        except:
            im = cv2.imread(im_fn)
    else:
        im = cv2.imread(im_fn)
    # global dur
    # dur += time.time() - start
    # return im, 0, 0

    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    intersection_threld = 1
    # 检查越界

    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys = augmentation(im, text_polys, scales, degrees, input_size)

    intersection_threld *= im.shape[0] / h
    h, w, _ = im.shape
    short_edge = min(h, w)
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    intersection_threld *= im.shape[0] / h
    h, w, _ = im.shape

    # normal images
    if config.img_norm:
        im = im.astype(np.float32)
        im /= 255.0
        im -= np.array((0.485, 0.456, 0.406))
        im /= np.array((0.229, 0.224, 0.225))

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):
        poly = poly.astype(np.int)
        if tag:
            cv2.fillPoly(training_mask, [poly], 0)

    #####################################
    # start = time.time()
    distance_map = get_distance_map_v3(text_polys, h, w, intersection_threld)
    # global dur
    # dur += time.time() - start

    ##############################
    imgs = data_aug.random_crop_author([im,training_mask, np.expand_dims(distance_map, 2)], (input_size, input_size))

    #return im, training_mask, distance_map
    return imgs[0], imgs[1], np.squeeze(imgs[2], 2)   #, time.time() - start   #im,training_mask#


def get_distance_map_v3(text_polys, h, w, intersection_threld):
    dist_map = np.zeros((h, w), dtype=np.float)
    #print('th: ', intersection_threld)
    inter_area_threld = 50 * intersection_threld
   # print('val: ', inter_area_threld)

    undraw_list = []
    for (i, text_poly) in enumerate(text_polys, start=1):
        if i in undraw_list:
            continue
        text_poly = text_poly.astype(np.int)
        p = Polygon.Polygon(text_poly)
        bOverlap = False
        for j, tempPoly in enumerate(text_polys, start=1):
            if j == i:
                continue
            p2 = Polygon.Polygon(tempPoly)
            if p.overlaps(p2):
                overlapP = p & p2
                if overlapP.area() > inter_area_threld:
                    undraw_list.append(j)
                    bOverlap = True
                    break
        if bOverlap:
            undraw_list.append(i)
        else:
            cv2.fillPoly(dist_map, [text_poly], i)


    for (idx, text_poly) in enumerate(text_polys, start=1):
        if not idx in undraw_list:
            continue
        temp_map = np.zeros((h, w), dtype=np.float)
        pts = []
        for pt in text_poly:
            pts.append([int(pt[0]), int(pt[1])])
        cv2.fillPoly(temp_map, np.array([pts]), idx)

        Intersection = np.where((dist_map > 0.01) & (temp_map > 0.01), 1, 0)
        Inter_count = np.sum(Intersection)

        if Inter_count < inter_area_threld:
            dist_map[temp_map == idx] = idx
            continue

        for j in undraw_list:
            inter = np.where((dist_map == j) & (temp_map > 0.01), 1, 0)
            inter_sum = np.sum(inter)
            if inter_sum == 0:       # 找出已绘制box中和当前box有交集的
                continue

            inter_region = np.where(dist_map==j, 1, 0)
            inter_region_sum = np.sum(inter_region)
            temp_map_sum = np.sum(temp_map)
            rate_temp = float(inter_sum) / temp_map_sum
            rate_inter_region = float(inter_sum) / inter_region_sum
            if rate_temp > rate_inter_region:
                dist_map[temp_map==idx] = idx
            else:
                dist_map[(temp_map==idx)&(inter!=1)] = idx
            if inter_sum == Inter_count:  # 当前box只与这个box相交
                break

    for (i, text_poly) in enumerate(text_polys, start=1):
        text_i = np.where(dist_map == i, 1, 0)
        text_i = text_i.astype(np.uint8)
        # 距离图
        distance_map_i = cv2.distanceTransform(text_i, distanceType=cv2.DIST_L2, maskSize=5)
        cv2.normalize(distance_map_i, distance_map_i, 0.3, 1, cv2.NORM_MINMAX, mask=text_i.astype(np.uint8))
        #dist_map[distance_map_i>0.1] = distance_map_i
        dist_map = np.where(distance_map_i>0.1, distance_map_i, dist_map)

        #dist_map = dist_map * (1 - text_i) + distance_map_i
    #
    # np.savetxt('F:\\distance_map.csv', distance_map, delimiter=',', fmt='%F')
    # input()
    return dist_map



#############################################################################
class IC15Dataset(data.Dataset):
    def __init__(self, data_dir, data_shape: int = 640, transform=None, target_transform=None):
        self.data_list = self.load_data(data_dir)
        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform

        #self.aug = augument()  #20200302增加新augument方式

    def __getitem__(self, index):
        img_path, text_polys, text_tags = self.data_list[index]
        img, training_mask, distance_map = image_label_v3(img_path, text_polys, text_tags,
                                                                   input_size=self.data_shape,
                                                                   scales = np.array(config.random_scales))

        #img = draw_bbox(img,text_polys)
        #img = self.aug(image=np.array(img))['image']  #20200302增加新augument方式

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            training_mask = self.target_transform(training_mask)

        return img, training_mask, distance_map

    def load_data(self, data_dir: str) -> list:
        data_list = []
        img_list = os.listdir(data_dir + '/img')
        for x in img_list:
        #for x in glob.glob(data_dir + '/img/*.jpg', recursive=True):
            d = pathlib.Path(x)
            label_path = os.path.join(data_dir, 'gt', ('gt_' + str(d.stem) + '.txt'))
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
                        label = params[8]
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

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img




from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



if __name__ == '__main__':
    import torch
    import config
    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    from collections import Counter

#F:\\imgs\\psenet_vis2s     F:\zzxs\dl-data\ICDAR\ICDAR2015\\train
    #F:\zzxs\dl-data\ICDAR\ICDAR2015\sample_IC15\\train
    train_data = IC15Dataset('../../data/IC15/test', data_shape=config.data_shape,
                           transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    empty_count = 0
    max_values = []

    # config.bd_loss = False

    time_sum = 0
    for i, (img, mask, distance_map) in enumerate(train_loader):
        pbar.update(1)
        # print(img.shape)  # BCWH
        # print(mask.shape)       #BWH
        #
        # print(distance_map.shape)  #BWH


        #print(dist_maps.shape)
        # print(label[0][-1].sum())
        # input()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dist_map", cv2.WINDOW_NORMAL)
        cv2.imshow('img', img.squeeze(0).numpy().transpose((1, 2, 0)))
        #cv2.imshow('mask', mask.numpy().transpose((1, 2, 0))*255)
        cv2.imshow('dist_map', distance_map.numpy().transpose((1, 2, 0)))
        cv2.waitKey()
        cv2.destroyAllWindows()

        #time_sum = time_sum + dur

    pbar.close()
    print('all time:', dur)
    print('count:', len(train_loader))
    print('ave time:', dur/len(train_loader))




    #     #dist_map
    #     if torch.sum(dist_maps) == 0:
    #         dist_maps = dist_maps.add(10)
    #
    #     dist_maps = torch.clamp(dist_maps, max=100)
    #
    #     dist_maps = dist_maps.numpy()
    #     if dist_maps.sum() == 0:
    #         empty_count = empty_count + 1
    #     max_value = dist_maps.max()
    #     if max_value != 0:
    #         max_values.append(max_value)
    #
    # temp = Counter(max_values)
    # print('empty_count: ', empty_count)
    # print('max in max: ', max(max_values))
    # print('lens in max: ', len(max_values))
    # print('max_values count: ', len(temp))
    # print('most_common: ', temp.most_common(10))
    # pbar.close()
    # # dist_map




    #print(count_dict)

        #np.savetxt('F:\\save_connect' + '.csv', label_list, delimiter=',', fmt='%d')
    #
    #     # for i in range(6):
    #     #     dist_map = dist_maps[0]
    #     #     y = np.array(dist_map[i])
    #     #     y = y.astype(np.int)
    #     #     # y = y.data.numpy()
    #     #     np.savetxt('F:\\imgs\\psenet_vis2s\\save_'+str(i)+'.csv', y, delimiter=',', fmt='%d')
    #
    #     # pbar.update(1)
    #
    #     #show_img((img[0] * mask[0].to(torch.float)).numpy().transpose(1, 2, 0), color=True)
    #     show_img(label[0])
    #     # show_img(mask[0])
    #     plt.show()
    #
    # pbar.close()

