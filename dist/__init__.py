import subprocess
import os
import numpy as np
import cv2
import torch
import timeit
import matplotlib.pyplot as plt

import config

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile dis: {}'.format(BASE_DIR))

from .dist import dist_cpp



## fast post propress in python-------------dilate_alg
def dilate_alg(center, region, biregion, center_area_th, full_area_th, full_min_area, center_min_area=5):
    # center = np.array(center)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))  # 椭圆结构
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 椭圆结构

    label_num, label_img = cv2.connectedComponents(center, connectivity=4)   #.astype(np.uint8)

    label_values = []
    for label_idx in range(1, label_num):
        score_i = np.mean(biregion[label_img == label_idx])  # cenver ave score
        if np.sum(label_img == label_idx) < center_min_area or score_i < center_area_th:
            label_img[label_img == label_idx] = 0
            continue

        label_i = np.where(label_img == label_idx, 255, 0)
        label_dilation = cv2.dilate(label_i.astype(np.uint8), kernel)

        score_i = np.mean(biregion[label_dilation == 255])  # region ave score
        area = np.sum(label_dilation == 255)
        if np.sum(label_dilation == 255) < 50 or score_i < full_area_th: # full_min_area
            label_img[label_dilation == 255] = 0
            continue

        # bi_label_dilation = np.where(label_dilation == 255, 0, 1)
        # label_dilation = np.where(label_dilation == 255, label_idx, 0)
        # label_img = bi_label_dilation * label_img + label_dilation
        #### label_img = np.where(label_dilation == 255, label_idx, 0)
        label_img[label_dilation == 255] = label_idx
        label_values.append(label_idx)

    return np.array(label_img), label_values


def decode(preds, scale):  
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 图片放大倍率
    :return: 最后的输出图和文本框
    """

    bi_region = preds[1, :, :]
    preds = preds[0, :, :]

    bi_region = torch.sigmoid(bi_region)
    if len(bi_region.shape) == 3:
        bi_region = bi_region.squeeze(0)

    preds = torch.sigmoid(preds)

    if len(preds.shape) == 3:
        preds = preds.squeeze(0)

    #
    preds = torch.add(preds, bi_region)
    preds = torch.add(preds, -1)
    #preds = preds + bi_region - 1

    # region = preds >= 0.295
    # center = preds >= 0.56
    ones_tensor = torch.ones_like(preds, dtype=torch.float32)
    zeros_tensor = torch.zeros_like(preds, dtype=torch.float32)

    # plt.imshow(preds.cpu().numpy() * 255)
    # plt.show()
    region = torch.where(preds >= 0.295, ones_tensor, zeros_tensor)  # 17:0.285   15:0.295
    # plt.imshow(region.cpu().numpy() * 255)
    # plt.show()

    center = torch.where(preds >= 0.6, ones_tensor, zeros_tensor)   # 17:0.54   15:0.64


    region = region.to(device='cpu', non_blocking=False).numpy()
    center = center.to(device='cpu', non_blocking=False).numpy()
    bi_region = bi_region.to(device='cpu', non_blocking=False).numpy()

    area_threld = int(250*scale)
    #17: 0.91, 0.98, 250
    #15: 0.95, 0.988, 250   extData:0.95,0.976

    pred = dist_cpp(center.astype(np.uint8), region.astype(np.uint8), bi_region, 0.93, 0.95, area_threld) # 93, 97

    # label_num, label_img = cv2.connectedComponents(pred.astype(np.uint8), connectivity=4)
    # print('label_num: ', label_num)


    bbox_list = []
    scores_list = []
    label_values = int(np.max(pred))
    for label_value in range(label_values+1):   # range(label_values+1)
        if label_value == 0:
            continue

        # score = np.where(pred == label_value, preds, 0)
        # score = np.mean(score)
        scores_list.append(1)

        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]
        rect = cv2.minAreaRect(points)
        # if rect[1][0] > rect[1][1]:
        #     if rect[1][1] <= 10*scale:
        #         continue
        # else:
        #     if rect[1][0] <= 10*scale:
        #         continue

        bbox = cv2.boxPoints(rect)
        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])

    return pred, np.array(bbox_list), scores_list  # , preds


def decode_curve(preds, scale):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :return: 最后的输出图和文本框
    """
    #
    bi_region = preds[1, :, :]
    preds = preds[0, :, :]
    bi_region = torch.sigmoid(bi_region)
    if len(bi_region.shape) == 3:
        bi_region = bi_region.squeeze(0)

    preds = torch.sigmoid(preds)

    if len(preds.shape) == 3:
        preds = preds.squeeze(0)

    #
    preds = torch.add(preds, bi_region)
    preds = torch.add(preds, -1)

    # plt.imshow(preds.cpu().numpy()*255)
    # plt.show()

    ones_tensor = torch.ones_like(preds, dtype=torch.float32)
    zeros_tensor = torch.zeros_like(preds, dtype=torch.float32)

    #CTW
    # region = torch.where(preds >= 0.295, ones_tensor, zeros_tensor)
    # center = torch.where(preds >= 0.6, ones_tensor, zeros_tensor)

    #Total
    region = torch.where(preds >= 0.285, ones_tensor, zeros_tensor)  # 0.285
    center = torch.where(preds >= 0.56, ones_tensor, zeros_tensor)   # 0.62

    region = region.to(device='cpu', non_blocking=False).numpy()
    center = center.to(device='cpu', non_blocking=False).numpy()
    bi_region = bi_region.to(device='cpu', non_blocking=False).numpy()

    # plt.imshow(region * 255)
    # plt.show()
    # plt.imshow(center * 255)
    # plt.show()

    #CTW
    # area_threld = int(180 * scale)
    # pred = dist_cpp(center.astype(np.uint8), region.astype(np.uint8), bi_region, 0.93, 0.972, area_threld)

    #Total
    area_threld = int(250 * scale)
    pred = dist_cpp(center.astype(np.uint8), region.astype(np.uint8), bi_region, 0.93, 0.97, area_threld)

    bbox_list = []
    label_values = int(np.max(pred))
    for label_value in range(label_values+1):   # range(label_values+1)
        if label_value == 0:
            continue

        # points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]
        # rect = cv2.minAreaRect(points)
        # if rect[1][0] > rect[1][1]:
        #     if rect[1][1] <= 10:
        #         continue
        # else:
        #     if rect[1][0] <= 10:
        #         continue

        binary = np.zeros(pred.shape, dtype='uint8')
        binary[pred == label_value] = 1

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        bbox = contour

        if bbox.shape[0] <= 2:
            continue

        bbox = bbox*1.0 / scale
        bbox = bbox.astype('int32')
        bbox_list.append(bbox.reshape(-1))
    return pred, bbox_list  # , preds

import timeit
from numba import jit, set_num_threads, prange
from numba.typed import List
set_num_threads(8)

@jit(parallel=True, nopython=True, cache=True)
def fast_decode_helper1(region_label_img, center_label_img, lab):
    # res = 0
    # count = 0
    # for i in prange(center_label_img.shape[0]):
    #     for j in range(center_label_img.shape[1]):
    #         if center_label_img[i][j] == lab:
    #             res += region_label_img[i][j]
    #             count += 1
    # if count == 0:
    #     return 0
    # return res//count

    pos = np.where(center_label_img == lab)
    num = len(pos[0])
    return region_label_img[pos[0][num//2]][pos[1][num//2]]
    # return region_label_img[pos[0][0]][pos[1][0]]

@jit(parallel=True, nopython=True, cache=True)
def fast_decode_helper2(isolated, region_label_img, center_label_img, center, region, bi_region, region_iso, center_iso,
                        min_center=0.93, min_region=0.95, min_region_area=250):
    region_ave = 0.0
    region_num = 0
    center_ave = 0.0
    center_num = 0
    isolated2 = isolated.copy()
    for i in prange(center_label_img.shape[0]):
        for j in range(center_label_img.shape[1]):
            if region_label_img[i][j] == 0:
                continue
            for k in range(len(region_iso)):
                if region_label_img[i][j] == region_iso[k]:
                    region_ave += bi_region[i][j]
                    region_num += 1
                    region[i][j] = 0
                    isolated2[i][j] = 255
            for k in range(len(center_iso)):
                if center_label_img[i][j] == center_iso[k]:
                    center_ave += bi_region[i][j]
                    center_num += 1
                    center[i][j] = 0
    if region_num == 0 or center_num == 0:
        return isolated, center, region
    region_ave /= region_num
    center_ave /= center_num
    if region_num > min_region_area and region_ave >= min_region and center_ave >= min_center:
        return isolated2, center, region
    else:
        return isolated, center, region

def fast_decode_curve(preds, scale):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :return: 最后的输出图和文本框
    """
    #
    bi_region = preds[1, :, :]
    preds = preds[0, :, :]
    bi_region = torch.sigmoid(bi_region)
    if len(bi_region.shape) == 3:
        bi_region = bi_region.squeeze(0)

    preds = torch.sigmoid(preds)

    if len(preds.shape) == 3:
        preds = preds.squeeze(0)

    #
    preds = torch.add(preds, bi_region)
    preds = torch.add(preds, -1)

    # plt.imshow(preds.cpu().numpy()*255)
    # plt.show()

    ones_tensor = torch.ones_like(preds, dtype=torch.float32)
    zeros_tensor = torch.zeros_like(preds, dtype=torch.float32)

    #CTW
    # region = torch.where(preds >= 0.295, ones_tensor, zeros_tensor)
    # center = torch.where(preds >= 0.6, ones_tensor, zeros_tensor)

    #Total
    region = torch.where(preds >= 0.285, ones_tensor, zeros_tensor)  # 0.285
    center = torch.where(preds >= 0.56, ones_tensor, zeros_tensor)   # 0.62

    region = region.to(device='cpu', non_blocking=False).numpy().astype(np.uint8)
    center = center.to(device='cpu', non_blocking=False).numpy().astype(np.uint8)
    bi_region = bi_region.to(device='cpu', non_blocking=False).numpy()

    # plt.imshow(region * 255)
    # plt.show()
    # plt.imshow(center * 255)
    # plt.show()

    t1 = timeit.default_timer()
    center_label_num, center_label_img = cv2.connectedComponents(center, connectivity=8)
    region_label_num, region_label_img = cv2.connectedComponents(region, connectivity=8)
    t1 = timeit.default_timer() - t1
    # print('t0: ', t1)

    bbox_list = []

    t1 = timeit.default_timer()
    # 孤立区域直接输出box
    intersect_labels = []
    for i in range(region_label_num+1):
        intersect_labels.append([])
    for lab in range(1, center_label_num, 1):
        # center_value = region_label_img[np.where(center_label_img==lab)][0]
        center_value = fast_decode_helper1(region_label_img, center_label_img, lab)
        if center_value <= region_label_num and center_value > 0:
            intersect_labels[center_value].append(lab)
    t1 = timeit.default_timer() - t1
    # print('t1: ', t1)

    # print(intersect_labels)
    t2 = timeit.default_timer()
    isolated = np.zeros(region.shape, dtype='uint8')
    all_isolate = True
    region_iso = List()
    center_iso = List()
    for i, region_labs in enumerate(intersect_labels):
        if len(region_labs) == 1:   # 孤立区域
            region_iso.append(i)
            center_iso.append(region_labs[0])

            if all_isolate:
                all_isolate = False

    if len(region_iso) > 0:
        isolated, center, region = fast_decode_helper2(isolated, region_label_img, center_label_img, center, region,
                                                   bi_region, region_iso, center_iso, 0.93, 0.97, 250 * scale)

    t2 = timeit.default_timer() - t2
    # print('t2: ', t2)

    isolated_contours, _ = cv2.findContours(isolated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in isolated_contours:
        bbox = contour
        if bbox.shape[0] <= 2:
            continue
        bbox = bbox * 1.0 / scale
        bbox = bbox.astype('int32')
        bbox_list.append(bbox.reshape(-1))
    if all_isolate:
        return isolated, bbox_list

    # 有连接区域继续处理
    #CTW
    # area_threld = int(180 * scale)
    # pred = dist_cpp(center.astype(np.uint8), region.astype(np.uint8), bi_region, 0.93, 0.972, area_threld)

    #Total
    t3 = timeit.default_timer()
    area_threld = int(250 * scale)
    pred = dist_cpp(center, region, bi_region, 0.93, 0.97, area_threld)
    t3 = timeit.default_timer() - t3
    # print('t3: ', t3)
    # print('iso/pt: ', t1+t2, '//', t3)
    # print(t1+t2, '//', t3)


    label_values = int(np.max(pred))
    for label_value in range(label_values+1):   # range(label_values+1)
        if label_value == 0:
            continue

        # points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]
        # rect = cv2.minAreaRect(points)
        # if rect[1][0] > rect[1][1]:
        #     if rect[1][1] <= 10:
        #         continue
        # else:
        #     if rect[1][0] <= 10:
        #         continue

        binary = np.zeros(pred.shape, dtype='uint8')
        binary[pred == label_value] = 1

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        bbox = contour
        if bbox.shape[0] <= 2:
            continue

        bbox = bbox*1.0 / scale
        bbox = bbox.astype('int32')
        bbox_list.append(bbox.reshape(-1))
    return pred, bbox_list  # , preds

def decode_biregion(preds, scale):
    preds = preds[0, :, :]
    preds = torch.sigmoid(preds)

    if len(preds.shape) == 3:
        preds = preds.squeeze(0)

    preds = preds > 0.93
    preds = preds.to(device='cpu', non_blocking=True).numpy()

    label_num, label_img = cv2.connectedComponents(preds.astype(np.uint8), connectivity=4)

    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label_img == label_idx) < 10:
            label_img[label_img == label_idx] = 0
            continue

        score_i = np.mean(preds[label_img == label_idx])
        if score_i < 0.93:
            continue
        label_values.append(label_idx)

    bbox_list = []
    scores_list = []
    for label_value in label_values:
        if label_value == 0:
            continue
        points = np.array(np.where(label_img == label_value)).transpose((1, 0))[:, ::-1]


        # score = np.where(pred == label_value, preds, 0)
        # score = np.mean(score)
        scores_list.append(1)

        if len(points) < 100 * scale:
            continue

        rect = cv2.minAreaRect(points)
        # if rect[1][0] > rect[1][1]:
        #     if rect[1][1] <= 10*scale:
        #         continue
        # else:
        #     if rect[1][0] <= 10*scale:
        #         continue

        bbox = cv2.boxPoints(rect)
        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])

    return preds, np.array(bbox_list)  #, scores_list  # , preds



def decode_curve_biregion(preds, scale):
    preds = preds[0, :, :]
    preds = torch.sigmoid(preds)

    if len(preds.shape) == 3:
        preds = preds.squeeze(0)

    preds = preds > 0.7
    preds = preds.to(device='cpu', non_blocking=True).numpy()

    label_num, label_img = cv2.connectedComponents(preds.astype(np.uint8), connectivity=4)

    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label_img == label_idx) < 10:
            label_img[label_img == label_idx] = 0
            continue

        score_i = np.mean(preds[label_img == label_idx])
        if score_i < 0.98:
            continue
        label_values.append(label_idx)

    bbox_list = []
    scores_list = []
    for label_value in label_values:
        if label_value == 0:
            continue
        points = np.array(np.where(label_img == label_value)).transpose((1, 0))[:, ::-1]

        # score = np.where(pred == label_value, preds, 0)
        # score = np.mean(score)
        scores_list.append(1)

        if len(points) < 100 * scale:
            continue

        binary = np.zeros(label_img.shape, dtype='uint8')
        binary[label_img == label_value] = 1

        _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        bbox = contour

        if bbox.shape[0] <= 2:
            continue

        bbox = bbox * 1.0 / scale
        bbox = bbox.astype('int32')
        bbox_list.append(bbox.reshape(-1))
    return label_img, bbox_list  # , preds


def decode_dist(preds, scale):  # origin=0.7311
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :return: 最后的输出图和文本框
    """

    preds = preds[0, :, :]
    preds = torch.sigmoid(preds)

    if len(preds.shape) == 3:
        preds = preds.squeeze(0)

    # region = preds >= 0.295
    # center = preds >= 0.56
    ones_tensor = torch.ones_like(preds, dtype=torch.float32)
    zeros_tensor = torch.zeros_like(preds, dtype=torch.float32)
    region = torch.where(preds >= 0.295, ones_tensor, zeros_tensor)
    center = torch.where(preds >= 0.56, ones_tensor, zeros_tensor)

    region = region.to(device='cpu', non_blocking=True).numpy()
    center = center.to(device='cpu', non_blocking=True).numpy()
    preds = preds.to(device='cpu', non_blocking=True).numpy()


    #
    # preds2 = preds > 0.8
    # cv2.imwrite('../save_dist_10_8.jpg', preds2 * 255)
    # preds2 = preds > 0.7
    # cv2.imwrite('../save_dist_10_7.jpg', preds2 * 255)
    # preds2 = preds > 0.56
    # cv2.imwrite('../save_dist_10_56.jpg', preds2 * 255)
    # preds2 = preds > 0.295
    # cv2.imwrite('../save_dist_10_29.jpg', preds2 * 255)
    # bi_region2 = bi_region > 0.9
    # cv2.imwrite('../save_bi_10.jpg', bi_region2 * 255)

    # pred2 = np.where((preds>=0.295) & (preds<=0.56), 1, 0)
    # cv2.imwrite('../region.jpg', region * 255)
    # cv2.imwrite('../center.jpg', center * 255)
    #
    # print('finish')
    # input()


    area_threld = int(250*scale)
    # print('in cpp')
    pred = dist_cpp(center.astype(np.uint8), region.astype(np.uint8), preds, 0.6, 0.6, area_threld)   # 0.95, 0.988, 200
    # plt.imshow(pred)
    # plt.show()

    # label_num, label_img = cv2.connectedComponents(pred.astype(np.uint8), connectivity=4)
    # print('label_num: ', label_num)
    #
    # plt.imshow(label_img)
    # plt.show()
    # cv2.imwrite('/home/beidou/pred.jpg', pred*30)
    #
    # print('out cpp')
    # input()


    bbox_list = []
    scores_list = []
    label_values = int(np.max(pred))
    for label_value in range(label_values+1):
        if label_value == 0:
            continue
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]


        # score = np.where(pred == label_value, preds, 0)
        # score = np.mean(score)
        scores_list.append(1)

        rect = cv2.minAreaRect(points)
        # if rect[1][0] > rect[1][1]:
        #     if rect[1][1] <= 10*scale:
        #         continue
        # else:
        #     if rect[1][0] <= 10*scale:
        #         continue

        bbox = cv2.boxPoints(rect)
        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])

    return pred, np.array(bbox_list), scores_list  # , preds


if __name__ == '__main__':
    logits = torch.tensor([[[0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.7, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.5, 0.5, 0, 0, 0, 0]],
                           [[0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.7, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.5, 0.5, 0, 0, 0, 0]]])

    preds, boxes_list = decode(logits, 1)
    print('preds:', preds)
    print('boxes_list:', boxes_list)

    # import matplotlib.pyplot as plt
    # from utils import draw_bbox
    # img = cv2.imread('F:\\img_pred14.jpg', cv2.IMREAD_GRAYSCALE)
    #
    # preds, boxes_list = decode(img, 1)
    # plt.imshow(preds)
    # plt.show()
    # h = 720
    # w = 1280
    # scale = 1900 / max(h, w)
    #
    # res = draw_bbox('F:\\img_14.jpg', np.array(boxes_list) / scale)
    # #cv2.imwrite('F:\\img_10_save_charm.jpg', res)
    # plt.imshow(res)
    # plt.show()


