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

from boundary_loss import class2one_hot, one_hot2dist
import Polygon
from Polygon.Utils import pointList

from albumentations import (
    Compose, RGBShift, RandomBrightness, RandomContrast,
    HueSaturationValue, ChannelShuffle, CLAHE,
    RandomContrast, Blur, ToGray, JpegCompression,
    CoarseDropout
)

data_aug = PSEDataAugment()

def augument():
    augm = Compose([
        RGBShift(),
        RandomBrightness(),
        RandomContrast(),
        HueSaturationValue(p=0.2),
        ChannelShuffle(),
        CLAHE(),
        Blur(),
        ToGray(),
        CoarseDropout()
    ],
    p=0.5)
    return augm

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


def generate_rbox(im_size, text_polys, text_tags, training_mask, i, n, m, origin_shrink=True):
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

        if origin_shrink:
            if n == 1:
                cv2.fillPoly(score_map, [poly], 1)
                if tag:
                    cv2.fillPoly(training_mask, [poly], 0)
            else:
                r_i = 1 - (1 - m) * (n - i) / (n - 1)
                # print('r_i:', r_i)
                d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
                pco = pyclipper.PyclipperOffset()
                # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
                pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked_poly = np.array(pco.Execute(-d_i))
                cv2.fillPoly(score_map, shrinked_poly, 1)
                # 制作mask
                # rect = cv2.minAreaRect(shrinked_poly)
                # poly_h, poly_w = rect[1]

                # if min(poly_h, poly_w) < 10:
                #     cv2.fillPoly(training_mask, shrinked_poly, 0)
                if tag:
                    cv2.fillPoly(training_mask, shrinked_poly, 0)
            # 闭运算填充内部小框
            # kernel = np.ones((3, 3), np.uint8)
            # score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel)
        else:
            if n != 2:
                raise ValueError('In my label schedual, n must be 2.')
            import Polygon
            from Polygon.Utils import pointList

            p = Polygon(poly)
            p.scale(m, m)
            pts = pointList(p)

            pts2 = []
            pts3 = []
            for pt in pts:
                pts2.append([int(pt[0]), int(pt[1])])
            #pts3.append(pts2)
            shrinked_poly=np.array([pts2])
            cv2.fillPoly(score_map, shrinked_poly, 1)
            if tag:
                cv2.fillPoly(training_mask, shrinked_poly, 0)

    return score_map, training_mask


def augmentation(im: object, text_polys: object, scales: object, degrees: object, input_size: object) -> object:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    if 'resize' in config.augment_list:
        im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5 and 'flip' in config.augment_list:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5 and 'rotate' in config.augment_list:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    # 640 × 640 random samples are cropped from the transformed images
    # im, text_polys = data_aug.random_crop_img_bboxes(im, text_polys)

    # im, text_polys = data_aug.resize(im, text_polys, input_size, keep_ratio=False)
    # im, text_polys = data_aug.random_crop_image_pse(im, text_polys, input_size)

    return im, text_polys


def image_label(im_fn: str, text_polys: np.ndarray, text_tags: list, n: int, m: float, input_size: int,
                degrees: int = 10,
                scales: np.ndarray = np.array([0.5, 1, 2.0, 3.0])) -> tuple:
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
    im, text_polys, = augmentation(im, text_polys, scales, degrees, input_size)

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
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, i, n, m, origin_shrink=config.origin_shrink)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)

    # cv2.imshow('score', score_maps.transpose((1, 2, 0)))


    kernel_mask = np.ones((h, w), dtype=np.uint8)
    kernel_map, kernel_mask = generate_rbox((h, w), text_polys, text_tags, kernel_mask, 0, 6, 0.5, origin_shrink=True)
    kernel_map = np.array([kernel_map], dtype=np.float32)

    # cv2.imshow('kernel_map', kernel_map.transpose((1, 2, 0)))
    # cv2.waitKey()


    ##############################
    #mid2 = time.time() - mid
    distance_map = get_distance_map(score_maps, overlap_map, score_maps_line)
    #dur = time.time() - start - mid2

    ##############################
    # imgs = data_aug.random_crop_author([im, score_maps.transpose((1, 2, 0)),training_mask, np.expand_dims(distance_map, 2)], (input_size, input_size))
    imgs = data_aug.random_crop_author([im, score_maps.transpose((1, 2, 0)), kernel_map.transpose((1, 2, 0)), kernel_mask, training_mask
                                           , np.expand_dims(distance_map, 2)], (input_size, input_size))

    #return im,score_maps,training_mask, distance_map, dur
    print(imgs[1].shape)
    score_maps = np.squeeze(imgs[1],2)
    print(imgs[2].shape)
    score_maps = np.squeeze(imgs[2], 2)
    print(imgs[5].shape)
    score_maps = np.squeeze(imgs[5], 2)
    input()

    return imgs[0], np.squeeze(imgs[1],2), np.squeeze(imgs[2],2), imgs[3], imgs[4], np.squeeze(imgs[5], 2)   #, dur   #im,score_maps,training_mask#


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
        cv2.normalize(distance_inter_map, distance_inter_map, 0.3, 1, cv2.NORM_MINMAX, mask=lab_img_i.astype(np.uint8))

    #相交区域
    score_maps_line2 = np.where(overlap_map == 1, 0.2, 0)
    distance_map = (1 - overlap_map) * distance_inter_map + score_maps_line2

    #边界
    score_maps_line2 = np.where(score_maps_line == 1, 0.2, 0)
    distance_map = (1 - score_maps_line) * distance_map + score_maps_line2


    # np.savetxt('F:\\distance_map_v10.csv', distance_map, delimiter=',', fmt='%F')
    # input()
    return distance_map


class PSEDataset_bd(data.Dataset):
    def __init__(self, data_dir, data_shape: int = 640, n=6, m=0.5, transform=None, target_transform=None):
        self.data_list = self.load_data(data_dir)
        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform
        self.n = n
        self.m = m

        self.aug = augument()  #20200302增加新augument方式

    def __getitem__(self, index):
        # print(self.image_list[index])
        img_path, text_polys, text_tags = self.data_list[index]
        img, score_maps, kernal_map, kernal_mask, training_mask, distance_map = image_label(img_path, text_polys, text_tags, input_size=self.data_shape,
                                                     n=self.n,
                                                     m=self.m,
                                                     scales = np.array(config.random_scales))
        #img = draw_bbox(img,text_polys)

        #img = self.aug(image=np.array(img))['image']  #20200302增加新augument方式

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_maps = self.target_transform(score_maps)
            training_mask = self.target_transform(training_mask)

        dist_loss_map = 0
        if config.bd_loss:
            # boundary loss
            score_maps_tensor = torch.tensor(score_maps).type(torch.int)
            one_hot_label = class2one_hot(score_maps_tensor, 2)  # c*2wh
            dist_maps_list = []
            for i in range(one_hot_label.shape[0]):  # c
                one_hot_label_i = one_hot_label[i].numpy()  # 2*wh
                dist_map = one_hot2dist(one_hot_label_i)  # 2*wh
                dist_map = dist_map[1, ...]  # 前景 wh
                dist_maps_list.append(dist_map)
            dist_loss_map = np.stack(dist_maps_list, axis=0)  # cwh

        return img, kernal_map, kernal_mask, training_mask, distance_map, dist_loss_map

    def load_data(self, data_dir: str) -> list:
        data_list = []
        for x in glob.glob(data_dir + '/img/*.jpg', recursive=True):
            d = pathlib.Path(x)
            label_path = os.path.join(data_dir, 'gt', ('gt_' + str(d.stem) + '.txt'))
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
    train_data = PSEDataset_bd('F:\zzxs\Experiments\dl-data\ICDAR\ICDAR2015\\train', data_shape=config.data_shape, n=1, m=config.m,
                           transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    empty_count = 0
    max_values = []

    # config.bd_loss = False

    time_sum = 0
    dur = 0

    #distance_map   torch.Size([1, 1, 640, 640])
    #dist_loss_map  torch.Size([1, 640, 640])
    #mask  torch.Size([1, 640, 640])
    for i, (img, kernel_map, kernal_mask, mask, distance_map, dist_loss_map) in enumerate(train_loader):

        pbar.update(1)
        print(dist_loss_map.shape)
        print(distance_map.shape)
        print(mask.shape)
        print(kernal_mask.shape)
        print(kernel_map.shape)
        kernal_mask = kernal_mask.numpy()
        print(np.sum(kernal_mask))
        mask = mask.numpy()
        print(np.sum(mask))
        input()

        # print(distance_map.shape)  #BWH
        #
        # # print(label[:, 0, :, :].shape)
        # print(img.shape)        #BCWH
        # #print(dist_maps.shape)
        # print(label.shape)      #BWH
        # # print(label[0][-1].sum())
        # print(mask.shape)       #BWH
        # input()
        cv2.imshow('dist_map', distance_map.numpy().transpose((1, 2, 0)))
        cv2.imshow('kernal_mask', kernal_mask.transpose((1, 2, 0))*255)
        cv2.imshow('mask', mask.transpose((1, 2, 0))*255)
        cv2.waitKey()
        cv2.destroyAllWindows()

        time_sum = time_sum + dur

    pbar.close()
    print('all time:', time_sum)
    print('count:', len(train_loader))
    print('ave time:', time_sum/len(train_loader))




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

