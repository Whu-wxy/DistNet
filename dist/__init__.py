import subprocess
import os
import numpy as np
import cv2
import torch

import config

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile dis: {}'.format(BASE_DIR))


def dist_warpper(region, center, probs=None):
    '''
    后处理在这里修改
    reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
    :return:
    '''
    from .dist import dist_cpp

    pred = dist_cpp(center.astype(np.uint8), region.astype(np.uint8), probs, 0.98, 0.9861)  #0.97, 0.978

    return np.array(pred)

##
def dilate_alg(center, min_area=5, probs=None):
    center = np.array(center)
    label_num, label_img = cv2.connectedComponents(center.astype(np.uint8), connectivity=4)

    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label_img == label_idx) < min_area:
            label_img[label_img == label_idx] = 0
            continue

        score_i = np.mean(probs[label_img == label_idx])  # 测试是否可以过滤难负样本
        if score_i < 0.85:
            continue
        label_values.append(label_idx)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))  # 椭圆结构
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 椭圆结构
    for label_idx in range(1, label_num):
        label_i = np.where(label_img == label_idx, 255, 0)
        label_dilation = cv2.dilate(label_i.astype(np.uint8), kernel)
        bi_label_dilation = np.where(label_dilation == 255, 0, 1)
        label_dilation = np.where(label_dilation == 255, label_idx, 0)
        label_img = bi_label_dilation * label_img + label_dilation

    return np.array(label_img), label_values


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

    region = preds >= 0.295
    center = preds >= 0.58  #config.max_threld

    #
    # plt.imshow(center)
    # plt.show()
    # plt.imshow(region)
    # plt.show()

    # pred, label_values = dilate_alg(center, min_area=5, probs=preds)
    pred = dist_warpper(region, center, bi_region)   #概率图改为传bi_region

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
    label_values = np.max(pred)
    for label_value in range(label_values+1):
        if label_value == 0:
            continue
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        # score_i = np.mean(bi_region[pred == label_value])   #在c代码中完成
        # if score_i < 0.95:
        #     continue

        if config.save_4_pt_box:
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)

            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        else:
            x, y, w, h = cv2.boundingRect(points)
            bbox_list.append([[x, y], [x + w, y + h]])
    return pred, np.array(bbox_list)  # , preds


if __name__ == '__main__':
    logits = torch.tensor([[[0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.7, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.5, 0.5, 0, 0, 0, 0]],
                           [[0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.7, 0.61, 0.3, 0],
                            [0.3, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0.3, 0.5, 0.5, 0, 0, 0, 0]]
                           ])
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


