# -*- coding: utf-8 -*-
# @Time    : 3/29/19 11:03 AM
# @Author  : zhoujun
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from boundary_loss import boundary_loss


class Loss(nn.Module):
    def __init__(self, Lambda, ratio=3, reduction='mean', bd_loss=False):
        """Implement PSE Loss.
        """
        super(Loss, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.Lambda = Lambda
        self.ratio = ratio
        self.reduction = reduction
        self.bd_loss = bd_loss

    def forward(self, outputs, labels, training_masks, bd_loss_weight=0, dist_maps=None):
        texts = outputs[:, -1, :, :]
        kernels = outputs[:, :-1, :, :]
        gt_texts = labels[:, -1, :, :]
        gt_kernels = labels[:, :-1, :, :]

        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        # full text dice loss with OHEM
        loss_text = self.dice_loss(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = selected_masks.to(outputs.device)
        kernels_num = gt_kernels.size()[1]

        # text kernals dice loss with OHEM
        for i in range(kernels_num):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.stack(loss_kernels).mean(0)
        if self.reduction == 'mean':
            loss_text = loss_text.mean()
            loss_kernels = loss_kernels.mean()
        elif self.reduction == 'sum':
            loss_text = loss_text.sum()
            loss_kernels = loss_kernels.sum()

        # boundary loss with OHEM
        if self.bd_loss:
            dist_texts = dist_maps[:, -1, :, :]
            dist_kernals = dist_maps[:, :-1, :, :]

            mask = selected_masks.unsqueeze(dim=1)
            bd_loss_text = self.boundary_loss_batch(texts.unsqueeze(dim=1), dist_texts.unsqueeze(dim=1))
            mask = mask.repeat(1, kernels_num, 1, 1)
            bd_loss_kernals = self.boundary_loss_batch(kernels, dist_kernals)

            # version 4
            # dice loss不加权重,20轮后开始加bd loss，不加权重
            # all losses
            # loss = self.Lambda * (loss_text + bd_loss_weight*bd_loss_text) + \
            #        (1 - self.Lambda) * (loss_kernels + bd_loss_weight*bd_loss_kernals)

            # version 5         
            # # 20轮后开始加bd loss，权重变化同v3 
            # all losses
            loss = self.Lambda * (bd_loss_weight*loss_text + (1-bd_loss_weight)*bd_loss_text) + \
                   (1 - self.Lambda) * (bd_loss_weight*loss_kernels + (1-bd_loss_weight)*bd_loss_kernals)
            
            # v6
            # loss = self.Lambda * (loss_text + bd_loss_weight*bd_loss_text) + \
            #        (1 - self.Lambda) * (loss_kernels + bd_loss_weight*bd_loss_kernals)

            # add BCE
            # full text dice loss with OHEM
            # bd_loss_text = self.BCE_Loss(texts, gt_texts, selected_masks)
            # # kernal text dice loss with OHEM
            # bce_kernels = []
            # for i in range(kernels_num):
            #     kernel_i = kernels[:, i, :, :]
            #     gt_kernel_i = gt_kernels[:, i, :, :]
            #     loss_kernel_i = self.BCE_Loss(kernel_i, gt_kernel_i, selected_masks)
            #     bce_kernels.append(loss_kernel_i)
            # bce_kernels = torch.stack(bce_kernels).mean(0)
            # bd_loss_kernals = bce_kernels.mean()

            
            return loss_text, loss_kernels, bd_loss_text, bd_loss_kernals, loss
        else:
            loss = self.Lambda * loss_text + (1 - self.Lambda) * loss_kernels
            return loss_text, loss_kernels, loss

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


if __name__ == '__main__':
    criteria = PSELoss(0.3)
    logits = torch.randn(1, 6, 4, 4)
    label = torch.randint(0, 2, (1, 6, 4, 4))
    print('label: ', label)

    loss = criteria(label.to(dtype=torch.float), label)
    loss2 = criteria(1 - label.to(dtype=torch.float) + 0.9, label)
    print('loss1: ', loss)
    print("loss2: ", loss2)