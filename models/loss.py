# -*- coding: utf-8 -*-
# @Time    : 3/29/19 11:03 AM
# @Author  : zhoujun
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from boundary_loss import boundary_loss


class Loss(nn.Module):
    def __init__(self, OHEM_ratio=3, reduction='mean'):
        """Implement PSE Loss.
        """
        super(Loss, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.OHEM_ratio = OHEM_ratio
        self.reduction = reduction

    def forward(self, output, label, training_masks, distance_map):

        selected_masks = self.ohem_batch(texts, label, training_masks)
        selected_masks = selected_masks.to(output.device)

        # full text dice loss with OHEM
        loss_text = self.dice_loss(output, label, selected_masks)

        if self.reduction == 'mean':
            loss_text = loss_text.mean()
        elif self.reduction == 'sum':
            loss_text = loss_text.sum()

        loss = loss_text
        return loss_text, loss


    def dice_loss(self, input, target, mask):
        # mask: 0:文本区域， 1：背景

        #增大前景部分权重,  越接近target，值越为1
        input = torch.sigmoid(input)

        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)
        # N*(C*W*H)

        # mask中1的部分保留(负样本分数较低的不学习)
        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001     #添加系数，防止为0
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
        # training_mask: 0:黑色,文本区域
        # gt_text：1：白色，文本区域

        if pos_num == 0:  # 无文本区域
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * self.ratio, neg_num))    #负样本最多是正样本点数三倍

        if neg_num == 0:  # gt_text全白（全是文本区域）
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        # kernal图，gt_text白色区域缩小
        neg_score = score[gt_text <= 0.5]               #背景
        neg_score_sorted = np.sort(-neg_score)          # 将负样本得分从高到低排序
        threshold = -neg_score_sorted[neg_num - 1]      #找出负样本分数最大的若干个像素
                        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks

    def boundary_loss_batch(self, score, dist_maps, mask=None):
        probs = torch.sigmoid(score)
        return boundary_loss(probs, dist_maps, mask)

    def BCE_Loss(self, scores, target, training_masks):
        scores = torch.sigmoid(scores)

        scores = scores.contiguous().view(scores.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)
        # N*(C*W*H)

        # mask中1的部分保留(负样本分数较低的不学习)
        scores = scores * mask
        target = target * mask
        return F.binary_cross_entropy(scores, target, reduction='mean')

    def weighted_regression(self, gaussian_map, gaussian_gt, training_mask):
        """
        Weighted MSE-loss
        Args:
            gaussian_map: gaussian_map from network outputs
            gaussian_gt: gt for gaussian_map
            training_mask:
        """
        gaussian_map = torch.sigmoid(gaussian_map)
        text_map = torch.where(gaussian_gt > 0.2, torch.ones_like(gaussian_gt), torch.zeros_like(gaussian_gt))
        center_map = torch.where(gaussian_gt > 0.7, torch.ones_like(gaussian_gt), torch.zeros_like(gaussian_gt))
        center_gt = torch.where(gaussian_gt > 0.7, gaussian_gt, torch.zeros_like(gaussian_gt))
        text_gt = torch.where(gaussian_gt > 0.2, gaussian_gt, torch.zeros_like(gaussian_gt))
        bg_map = 1. - text_map

        pos_num = torch.sum(text_map)
        neg_num = torch.sum(bg_map)

        pos_weight = neg_num * 1. / (pos_num + neg_num)
        neg_weight = 1. - pos_weight

        #     mse_loss = F.mse_loss(gaussian_map, gaussian_gt, reduce='none')
        mse_loss = F.smooth_l1_loss(gaussian_map, gaussian_gt, reduce='none')
        weighted_mse_loss = mse_loss * (text_map * pos_weight + bg_map * neg_weight) * training_mask

        center_region_loss = torch.sum(center_gt * mse_loss * training_mask) / center_gt.sum()
        #     border_loss = torch.sum(border_map * mse_loss * training_mask) / border_map.sum()

        return weighted_mse_loss.mean(), torch.sum(
            text_gt * mse_loss * training_mask) / text_map.sum(), center_region_loss


if __name__ == '__main__':
    criteria = Loss(0.3)
    logits = torch.randn(1, 6, 4, 4)
    label = torch.randint(0, 2, (1, 6, 4, 4))
    print('label: ', label)

    loss = criteria(label.to(dtype=torch.float), label)
    loss2 = criteria(1 - label.to(dtype=torch.float) + 0.9, label)
    print('loss1: ', loss)
    print("loss2: ", loss2)