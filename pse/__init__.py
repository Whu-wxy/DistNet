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

def pse_warpper(region, center, min_area=5):
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
        label_values.append(label_idx)

    pred = pse_cpp(label, region)
    return np.array(pred), label_values


def decode(preds, scale, threshold=config.decode_threld): #origin=0.7311
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    preds = torch.sigmoid(preds)
    if len(preds.shape) == 3:
        preds = preds.squeeze(0)
    preds = preds.detach().cpu().numpy()

    region = preds >= config.min_threld   #按阈值变为2值图
    center = preds >= config.max_threld  # 按阈值变为2值图
    pred, label_values = pse_warpper(region, center, 5)

    #pred, label_values = pse(region, center, 5)

    bbox_list = []
    for label_value in label_values:
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        # if points.shape[0] < 800 / (scale * scale):  #text区域点数
        #     continue


        # score_i = np.mean(score[pred == label_value])   #20200317 TO TEST!
        # if score_i < 0.9:  # 降低是否可以提高召回率？ 0.93
        #     continue

        if config.save_4_pt_box:
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)

            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        else:
            x, y, w, h = cv2.boundingRect(points)
            bbox_list.append([[x, y], [x+w, y+h]])
    return pred, np.array(bbox_list)


if __name__ == '__main__':
    logits = torch.tensor([[[0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0, 0.3, 0.61, 0.7, 0.61, 0.3, 0],
                            [0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                            [0, 0.5, 0.5, 0, 0, 0, 0]]])

    preds, boxes_list = decode(logits, 1)
    print('preds:', preds)
    print('boxes_list:', boxes_list)