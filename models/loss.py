# -*- coding: utf-8 -*-
# @Time    : 3/29/19 11:03 AM
# @Author  : zhoujun
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from boundary_loss import boundary_loss
import config

class Loss(nn.Module):
    def __init__(self, OHEM_ratio=3, reduction='mean'):
        """Implement PSE Loss.
        """
        super(Loss, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.OHEM_ratio = OHEM_ratio
        self.reduction = reduction


    def forward_region(self, output, label, output_region, region_lab, region_mask, training_masks, bd_loss_weight=0, dist_maps=None):

        selected_masks = self.ohem_batch(output, label, training_masks)
        selected_masks = selected_masks.to(output.device)
        region_mask = self.ohem_batch(output_region, label, region_mask)
        region_mask = region_mask.to(output.device)

        # full text dice loss with OHEM
        output = torch.sigmoid(output)

        output_region = torch.sigmoid(output_region)
        dice_full = self.dice_loss(output_region, region_lab, region_mask)

        center_gt = torch.where(label >= config.max_threld, label,
                                torch.zeros_like(label))
        region_map = torch.where(output >= config.min_threld, output, torch.zeros_like(output))
        center_map = torch.where(output >= config.max_threld, output, torch.zeros_like(output))

        dice_region = self.dice_loss(region_map, label, selected_masks)
        dice_center = self.dice_loss(center_map, center_gt, selected_masks)
        weighted_mse_region = self.weighted_regression(output, label, region_mask)  #有加权，不用OHEM的mask

        # boundary loss with OHEM
        if config.bd_loss:
            mask = training_masks.unsqueeze(dim=1)  #bchw
            bd_loss = self.boundary_loss_batch(region_map.unsqueeze(dim=1), dist_maps, mask)
            bd_loss = bd_loss_weight * bd_loss

            loss = dice_center + dice_region + weighted_mse_region + bd_loss
            return dice_center, dice_region, weighted_mse_region, bd_loss, loss
        else:
            loss = dice_center + dice_region + weighted_mse_region + dice_full

            return dice_center, dice_region, weighted_mse_region, dice_full, loss


    def forward(self, output, label, training_masks, bd_loss_weight=0, dist_maps=None):

        #
        output_bi_region = output[:, 1, :, :]
        output_bi_region = torch.sigmoid(output_bi_region)
       
        output = output[:, 0, :, :]
        #

        output = torch.sigmoid(output)

        center_gt = torch.where(label >= config.max_threld, label,
                                torch.zeros_like(label))

        region_map = torch.where(output >= config.min_threld, output, torch.zeros_like(output))
        center_map = torch.where(output >= config.max_threld, output, torch.zeros_like(output))

        #
        bi_region_gt = torch.where(label >= config.min_threld, torch.ones_like(label),
                                   torch.zeros_like(label))
        selected_masks = self.ohem_batch(output, bi_region_gt, training_masks)
        selected_masks = selected_masks.to(output.device)
        bir_selected_masks = self.ohem_batch(output_bi_region, bi_region_gt, training_masks)
        bir_selected_masks = bir_selected_masks.to(output.device)
        

        dice_bi_region = self.dice_loss(output_bi_region, bi_region_gt, bir_selected_masks)
        #

        dice_region = self.dice_loss(region_map, label, selected_masks)
        dice_center = self.dice_loss(center_map, center_gt, selected_masks)
        weighted_mse_region = self.weighted_regression(output, label, selected_masks)  #有加权，不用OHEM的mask

        # boundary loss with OHEM
        if config.bd_loss:
            mask = training_masks.unsqueeze(dim=1)  #bchw
            bd_loss = self.boundary_loss_batch(region_map.unsqueeze(dim=1), dist_maps, mask)
            bd_loss = bd_loss_weight * bd_loss

            loss = dice_center + dice_region + weighted_mse_region + bd_loss
            return dice_center, dice_region, weighted_mse_region, bd_loss, loss
        else:
            loss = dice_center + dice_region + weighted_mse_region + dice_bi_region

            return dice_center, dice_region, weighted_mse_region, loss, dice_bi_region


    def dice_loss(self, input, target, mask):
        # mask: 0:文本区域， 1：背景

        #增大前景部分权重,  越接近target，值越为1
        # input = torch.sigmoid(input)

        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)
        # N*(C*W*H)

        # mask中1的部分保留(负样本分数较低的不学习), ###从label中去除
        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001     #添加系数，防止为0
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)

        if self.reduction == 'mean':
            dice_loss = torch.mean(d)
        elif self.reduction == 'sum':
            dice_loss = torch.sum(d)

        return 1 - dice_loss

    def ohem_single(self, score, gt_text, training_mask):
        # label像素总数-有###的label像素数
        pos_num = (int)(np.sum(gt_text > config.min_threld)) - (int)(np.sum((gt_text > config.min_threld) & (training_mask <= 0.5)))
        # training_mask: 0:黑色,有tag(###)的区域标记为0
        # gt_text：1：白色，文本区域

        if pos_num == 0:  # 无文本区域
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= config.min_threld))   #背景像素数
        neg_num = (int)(min(pos_num * self.OHEM_ratio, neg_num))    #负样本最多是正样本点数三倍

        if neg_num == 0:  # gt_text全白（全是文本区域）
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        # kernal图，gt_text白色区域缩小
        neg_score = score[gt_text <= config.min_threld]               #背景
        neg_score_sorted = np.sort(-neg_score)          # 将负样本得分从高到低排序
        threshold = -neg_score_sorted[neg_num - 1]      #找出负样本分数最大的若干个像素
                        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((score >= threshold) | (gt_text > config.min_threld)) & (training_mask > 0.5)
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
        #probs = torch.sigmoid(score)
        return boundary_loss(score, dist_maps, mask)

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

    def weighted_regression(self, distance_map, distance_gt, training_mask):
        """
        Weighted MSE-loss
        Args:
            distance_map: distance_map from network outputs
            distance_gt: gt for distance_map
            training_mask:
        """
        # distance_gt = distance_gt * training_mask    # ###处为0

        text_gt = torch.where(distance_gt > config.min_threld, torch.ones_like(distance_gt), torch.zeros_like(distance_gt))
        bg_gt = 1. - text_gt

        pos_num = torch.sum(text_gt)
        neg_num = torch.sum(bg_gt)

        pos_weight = neg_num * 1. / (pos_num + neg_num)
        neg_weight = 1. - pos_weight
        #
        mse_loss = F.mse_loss(distance_map, distance_gt, reduction='mean')   #均方误差
        # #     mse_loss = F.smooth_l1_loss(distance_map, distance_gt, reduction='none')
        weighted_mse_loss = mse_loss * (text_gt * pos_weight + bg_gt * neg_weight)    # * training_mask

        return weighted_mse_loss.mean()


if __name__ == '__main__':
    criteria = Loss()

    logits = torch.tensor([[[0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                           [0, 0.3, 0.61, 0.7,  0.61, 0.3, 0],
                           [0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                           [0, 0.5,     0.5,    0, 0, 0, 0]]])

    label = torch.tensor([[[0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                           [0, 0.3, 0.61, 0.7,  0.61, 0.3, 0],
                           [0, 0.3, 0.61, 0.61, 0.61, 0.3, 0],
                           [0, 0,     0,    0, 0, 0, 0]]])

    label = torch.tensor([[[0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0]]])

    logits2 = torch.tensor([[[0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 0, 0, 1, 0],
                           [0, 1, 1, 0, 0, 0, 0]]])


    mask = torch.tensor(np.where(label > 0, 1, 0))
    dice_center, dice_region, weighted_mse_region, bd_loss, loss = criteria(logits2.to(dtype=torch.float), label.to(dtype=torch.float), mask, 1, )



    # dice_center2, dice_region2, weighted_mse_region2, loss2 = criteria(1 - label.to(dtype=torch.float) + 0.1, label, label)
    print('loss1: ', dice_center, dice_region, weighted_mse_region, loss)
    print('bd:', bd_loss)


    # print("loss2: ", dice_center2, dice_region2, weighted_mse_region2, loss2)
