import subprocess
import os
import numpy as np
import cv2
import torch

import config

from pse.pypse import pse

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

def pse_warpper(region, center, min_area=5, probs=None):
    '''
    后处理在这里修改
    reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
    :return:
    '''
    from .pse import pse_cpp

    center = np.array(center)
    label_num, label = cv2.connectedComponents(center.astype(np.uint8), connectivity=4) #C的代码从最小的kernel开始
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue

        score_i = np.mean(probs[label == label_idx])   #测试是否可以过滤难负样本
        if score_i < 0.97:
            continue

        #prob改为bi_region试试

        label_values.append(label_idx)

    pred = pse_cpp(label, region)

    return np.array(pred), label_values


def dilate_alg(center, min_area=5, probs=None):
    center = np.array(center)
    label_num, label_img = cv2.connectedComponents(center.astype(np.uint8), connectivity=4)

    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label_img == label_idx) < min_area:
            label_img[label_img == label_idx] = 0
            continue

        score_i = np.mean(probs[label_img == label_idx])  # 测试是否可以过滤难负样本
        if score_i < 0.66:
            continue
        label_values.append(label_idx)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))  # 椭圆结构
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 椭圆结构
    for label_idx in label_values:
        label_i = np.where(label_img == label_idx, 255, 0)
        label_dilation = cv2.dilate(label_i.astype(np.uint8), kernel)
        bi_label_dilation = np.where(label_dilation == 255, 0, 1)
        label_dilation = np.where(label_dilation == 255, label_idx, 0)
        label_img = bi_label_dilation * label_img + label_dilation

    return np.array(label_img), label_values


def decode_region(preds, scale, threshold=config.decode_threld): #origin=0.7311
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """

    preds_dist = preds[0, :, :]
    preds_region = preds[1, :, :]
    preds_dist = torch.sigmoid(preds_dist)
    preds_region = torch.sigmoid(preds_region)
    preds_region = preds_region.detach().cpu().numpy()

    #preds_dist = preds_dist + preds_region

    # if len(preds_dist.shape) == 3:
    #     preds_dist = preds_dist.squeeze(0)

    preds_dist = preds_dist.detach().cpu().numpy()

    # region = preds >= 77   #按阈值变为2值图
    # center = preds >= 160  # 按阈值变为2值图
    region = preds_dist >= 0.3
    center = preds_dist >= 0.75    #1.7
    # print(region)
    # input()
    #
    #
    # plt.imshow(center)
    # plt.show()
    # plt.imshow(region)
    # plt.show()

    #pred, label_values = dilate_alg(center)
    pred, label_values = pse_warpper(region, center, 5)
    #pred, label_values = pse(region, center, 5)

    # plt.imshow(pred)
    # plt.show()
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 椭圆结构
    # for label_idx in label_values:
    #     label_i = np.where(pred == label_idx, 255, 0)
    #     bi_label_i = np.where(label_i == 255, 0, 1)
    #     label_dilation = cv2.erode(cv2.dilate(label_i.astype(np.uint8), kernel), kernel)
    #     #label_dilation = cv2.dilate(label_i.astype(np.uint8), kernel)
    #
    #     label_dilation = np.where(label_dilation == 255, label_idx, 0)
    #     pred = bi_label_i * pred + label_dilation

    bbox_list = []
    for label_value in label_values:
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        # if points.shape[0] < 800 / (scale * scale):  #text区域点数
        #     continue

        score_i = np.mean(preds_region[pred == label_value])
        if score_i < 0.5:  # 降低是否可以提高召回率？ 0.93
            continue

        if config.save_4_pt_box:
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)

            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        else:
            x, y, w, h = cv2.boundingRect(points)
            bbox_list.append([[x, y], [x+w, y+h]])
    return pred, np.array(bbox_list)  #, preds

#_nokerkel
def decode(preds, scale, threshold=config.decode_threld):  # origin=0.7311
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """

    #
    bi_region = preds[1, :, :]
    preds = preds[0, :, :]
    bi_region = torch.sigmoid(bi_region)
    if len(bi_region.shape) == 3:
        bi_region = bi_region.squeeze(0)
    bi_region = bi_region.detach().cpu().numpy()

    #bi_region = bi_region>0.7311

    #
    #cv2.imwrite('../save.jpg', bi_region*255)
    #input()

    preds = torch.sigmoid(preds)

    if len(preds.shape) == 3:
        preds = preds.squeeze(0)
    preds = preds.detach().cpu().numpy()

    #
    preds = preds + bi_region - 1
    #

    region = preds >= 0.3
    center = preds >= 0.6  #config.max_threld

    #
    # plt.imshow(center)
    # plt.show()
    # plt.imshow(region)
    # plt.show()

    #pred, label_values = dilate_alg(center, min_area=5, probs=preds)
    pred, label_values = pse_warpper(region, center, 5, bi_region)   #概率图改为传bi_region
    # pred, label_values = pse(region, center, 5)

    # plt.imshow(pred)
    # plt.show()
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 椭圆结构
    # for label_idx in label_values:
    #     label_i = np.where(pred == label_idx, 255, 0)
    #     bi_label_i = np.where(label_i == 255, 0, 1)
    #     label_dilation = cv2.erode(cv2.dilate(label_i.astype(np.uint8), kernel), kernel)
    #     #label_dilation = cv2.dilate(label_i.astype(np.uint8), kernel)
    #
    #     label_dilation = np.where(label_dilation == 255, label_idx, 0)
    #     pred = bi_label_i * pred + label_dilation

    bbox_list = []
    for label_value in label_values:
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        score_i = np.mean(bi_region[pred == label_value])
        if score_i < 0.977:
            continue


        if config.save_4_pt_box:
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)

            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        else:
            x, y, w, h = cv2.boundingRect(points)
            bbox_list.append([[x, y], [x + w, y + h]])
    return pred, np.array(bbox_list)  # , preds


if __name__ == '__main__':
    # logits = torch.tensor([[[0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
    #                         [0, 0.3, 0.61, 0.7, 0.61, 0.3, 0],
    #                         [0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
    #                         [0, 0.5, 0.5, 0, 0, 0, 0]]])
    # preds, boxes_list = decode(logits, 1)
    # print('preds:', preds)
    # print('boxes_list:', boxes_list)

    import matplotlib.pyplot as plt
    from utils import draw_bbox
    img = cv2.imread('F:\\img_pred14.jpg', cv2.IMREAD_GRAYSCALE)

    preds, boxes_list = decode(img, 1)
    plt.imshow(preds)
    plt.show()
    h = 720
    w = 1280
    scale = 1900 / max(h, w)

    res = draw_bbox('F:\\img_14.jpg', np.array(boxes_list) / scale)
    #cv2.imwrite('F:\\img_10_save_charm.jpg', res)
    plt.imshow(res)
    plt.show()


