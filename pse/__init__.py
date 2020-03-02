import subprocess
import os
import numpy as np
import cv2
import torch

import config

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

def pse_warpper(kernals, min_area=5, origin_shrink=True):
    '''
    后处理在这里修改
    reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
    :param kernals:
    :param min_area:
    :return:
    '''
    from .pse import pse_cpp
    #from .pypse import pse
    kernal_num = len(kernals)
    if not kernal_num:
        return np.array([]), []
    kernals = np.array(kernals)
    label_num, label = cv2.connectedComponents(kernals[0].astype(np.uint8), connectivity=4) #从最小的kernel开始
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)

    if config.n == 1:                   #FPN baseline不进行pse
        return kernals, label_values
    else:
        if config.origin_shrink:
            pred = pse_cpp(label, kernals)   #, c=kernal_num
        else:
            pred = polygon_decode(label)

        return np.array(pred), label_values


def polygon_decode(label):  #连通域图
    # 轮廓提取
    pred = label
    label = label.astype(np.uint8)
    (contours, hier) = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []

    #面积排序
    for cont in contours:
        c = cont.tolist()
        c2 = [pt[0] for pt in c]
        poly = Polygon(c2)
        polys.append(poly)

    polys = sort_area(polys)

    center_list = []
    for p in polys:
        center_list.append(label[int(p.center()[1]), int(p.center()[0])])

    #多边形放大
    for i, p in enumerate(polys):
        p.scale(1/scale, 1/scale)
        pts = pointList(p)

        pts2 = []
        for pt in pts:
            pts2.append([int(pt[0]), int(pt[1])])

        back_poly = np.array([pts2])

        value = center_list[i]
        value = int(value)

        cv2.fillPoly(pred, back_poly, (value)) #用kernel中心点值进行填充
    return pred

def sort_area(polys):
    L = [(Polygon.area(poly), poly) for poly in polys]
    L = sorted(L, key=lambda x : x[0], reverse = True)
    L2 = [l[1] for l in L]
    return L2

def decode(preds, scale, threshold=config.decode_threld): #origin=0.7311
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    preds = torch.sigmoid(preds)
    preds = preds.detach().cpu().numpy()

    score = preds[-1].astype(np.float32)   #从小到大排列，取最大的mask
    preds = preds > threshold   #按阈值变为2值图
    # preds = preds * preds[-1] # 使用最大的kernel作为其他小图的mask,不使用的话效果更好
    pred, label_values = pse_warpper(preds, min_area=5, origin_shrink=config.origin_shrink)
    bbox_list = []
    for label_value in label_values:
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < 800 / (scale * scale):
            continue

        score_i = np.mean(score[pred == label_value])
        if score_i < 0.93:   #降低是否可以提高召回率？ 0.93
            continue

        if config.save_4_pt_box:
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)

            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        else:
            x, y, w, h = cv2.boundingRect(points)
            bbox_list.append([[x, y], [x+w, y+h]])
    return pred, np.array(bbox_list)
