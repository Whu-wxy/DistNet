import cv2
import os
import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import shutil
import time
import numpy as np
import torch
from tqdm import tqdm
import glob

from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from dataset.data_utils import DataLoaderX
from dataset.ReCTS import ReCTSDataset

from models import FPN_ResNet
from models.loss import Loss
# from models.fpn_resnet_atten_v1 import FPN_ResNet_atten_v1
# from models.fpn_resnet_atten_v2 import FPN_ResNet_atten_v2
from models.SA_FPN import SA_FPN

from utils.utils import load_checkpoint, save_checkpoint, setup_logger
#from dist import decode as dist_decode


from utils.radam import RAdam
from utils.ranger import Ranger
from utils.adabound import AdaBound, AdaBoundW
from utils.over9000 import Over9000
from utils.lamb import Lamb, log_lamb_rs
from utils.lamb_v3 import Lamb as Lamb_v3
from utils.la_lamb import La_Lamb, La_Lamb_v3

from boundary_loss import class2one_hot, one_hot2dist

from models.GFF_FPN import GFF_FPN
from models.resnet_FPEM import ResNet_FPEM
from models.craft import CRAFT

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# learning rate的warming up操作
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step, writer, logger):

    net.train()
    train_loss = 0.
    start = time.time()

    if scheduler == None:
        lr = optimizer.param_groups[-1]['lr']
    else:
        lr = scheduler.get_lr()[0]

    if config.if_warm_up:
        lr = adjust_learning_rate(optimizer, epoch)

    for i, (images, training_mask, distance_map) in enumerate(train_loader):
        cur_batch = images.size()[0]
        non_blocking = False
        if config.pin_memory and config.workers > 1:
            non_blocking = True

        #images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)
        images = images.to(device, non_blocking=non_blocking)

        # Forward
        outputs = net(images)   #B1HW

        # labels, training_mask后面放到gpu是否会占用更少一些显存？
        training_mask = training_mask.to(device, non_blocking=non_blocking)
        distance_map = distance_map.to(device, non_blocking=non_blocking)   #label
        distance_map = distance_map.to(torch.float)

        #outputs = torch.squeeze(outputs, dim=1)

        #
        dice_center, dice_region, weighted_mse_region, loss, dice_bi_region = criterion(outputs, distance_map, training_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        dice_center = dice_center.item()
        dice_region = dice_region.item()
        weighted_mse_region = weighted_mse_region.item()
        dice_bi_region = dice_bi_region.item()
        loss = loss.item()
        cur_step = epoch * all_step + i

        writer.add_scalar(tag='Train/dice_center', scalar_value=dice_center, global_step=cur_step)
        writer.add_scalar(tag='Train/dice_region', scalar_value=dice_region, global_step=cur_step)
        writer.add_scalar(tag='Train/dice_bi_region', scalar_value=dice_bi_region, global_step=cur_step)
        writer.add_scalar(tag='Train/weighted_mse_region', scalar_value=weighted_mse_region, global_step=cur_step)
        writer.add_scalar(tag='Train/loss', scalar_value=loss, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=lr, global_step=cur_step)

        batch_time = time.time() - start
        logger.info(
            '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, loss: {:.4f}, dice_center_loss: {:.4f}, dice_region_loss: {:.4f}, weighted_mse_region_loss: {:.4f}, dice_bi_region: {:.4f}, time:{:.4f}, lr:{}'.format(
                epoch, config.epochs, i, all_step, cur_step, cur_batch / batch_time, loss, dice_center, dice_region, weighted_mse_region, dice_bi_region, batch_time, lr))
        start = time.time()

        if cur_step == 500 or (cur_step % config.show_images_interval == 0 and  cur_step != 0):
            # show images on tensorboard
            if config.display_input_images:
                ######image
                x = vutils.make_grid(images.detach().cpu(), nrow=4, normalize=True, scale_each=True, padding=20)
                writer.add_image(tag='input/image', img_tensor=x, global_step=cur_step)
                ######distance_map
                show_distance_map = distance_map * training_mask
                show_distance_map = show_distance_map.detach().cpu()
                show_distance_map = show_distance_map[:8, :, :]
                show_distance_map = vutils.make_grid(show_distance_map.unsqueeze(1), nrow=4, normalize=False, padding=20,
                                              pad_value=1)
                writer.add_image(tag='input/distmap', img_tensor=show_distance_map, global_step=cur_step)

            if config.display_output_images:
                ######output
                outputs = outputs[:, 0, :, :]
                outputs = torch.sigmoid(outputs)
                show_y = outputs.detach().cpu()
                show_y = show_y[:8, :, :]
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=4, normalize=False, padding=20, pad_value=1)
                writer.add_image(tag='output/preds', img_tensor=show_y, global_step=cur_step)

    if scheduler!=None:
        scheduler.step()   #scheduler.step behind optimizer after pytorch1.1
    writer.add_scalar(tag='Train_epoch/loss', scalar_value=train_loss / all_step, global_step=epoch)
    return train_loss / all_step, lr


def main(model, criterion):

    if os.path.exists(config.output_dir):
        return

    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))
    logger.info(config.print())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    torch.set_default_tensor_type(torch.DoubleTensor)
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_data = ReCTSDataset(config.trainroot, data_shape=config.data_shape, transform=transforms.ToTensor(), ignore_english=True)
    # train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
    #                                num_workers=int(config.workers))

    train_loader = DataLoaderX(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=int(config.workers), pin_memory=config.pin_memory)

    writer = SummaryWriter(config.output_dir)

    # if not config.pretrained and not config.restart_training:
    #     model.apply(weights_init)
    #     logger.info('weights kaiming_normal init.')

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 600, 800).to(device))
    # writer.add_graph(models=models, input_to_model=dummy_input)

    if config.optim == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, momentum=0.99)
    elif config.optim == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    elif config.optim == 'adamw':
        optimizer = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    elif config.optim == 'radam':
        optimizer = RAdam([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim == 'ranger':
        optimizer = Ranger([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim == 'adabound':
        optimizer = AdaBound([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim == 'RangerLars':
        optimizer = Over9000([{'params': model.parameters(), 'initial_lr': config.lr}], alpha=0.5, k=6, lr=config.lr)
    elif config.optim == "lamb":
        optimizer = Lamb([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim == "lamb_v3":
        optimizer = Lamb_v3([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim == "la_lamb_v3":
        optimizer = La_Lamb_v3([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay
                                , alpha=0.5, k=6)
    elif config.optim == "la_lamb":
        optimizer = La_Lamb([{'params': model.parameters(), 'initial_lr': config.lr}], lr=config.lr, weight_decay=config.weight_decay
                                , alpha=0.5, k=6)
    else:
        raise ValueError('Chech optimizer setting.')

    scheduler = None
    if config.checkpoint != '':  # and not config.restart_training
        if config.load_lr:
            start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
            logger.info('model and optimizer load from checkpoint.')
        else:
            start_epoch = load_checkpoint(config.checkpoint, model, logger, device, None)
        if config.lr_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                         last_epoch=-1) #start_epoch   #gai wei chuan optim shiyishi///-1
        elif config.lr_scheduler == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.lr, max_lr=config.max_lr, mode=config.LR_mode,
                                                          step_size_up=config.step_size_up, gamma=config.lr_gamma, cycle_momentum=False, last_epoch=start_epoch)
        elif config.lr_scheduler == '':
            scheduler = None
    else:
        start_epoch = config.start_epoch
        if config.lr_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)
        elif config.lr_scheduler == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.lr, max_lr=config.max_lr, mode=config.LR_mode,
                                                          step_size_up=config.step_size_up, gamma=config.lr_gamma, cycle_momentum=False)
        elif config.lr_scheduler == '':
            scheduler = None

    if config.restart_training:
        start_epoch = config.start_epoch

    logger.info('begin from {} epoch.'.format(start_epoch))
    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    epoch = 0
    try:
        for epoch in range(start_epoch, config.epochs):
            start = time.time()
            train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                         writer, logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss, time.time() - start, lr))
            net_save_path = '{}/DistNet_ReCTS_{}_loss{:.6f}.pth'.format(config.output_dir, epoch,
                                                                                          train_loss)
            save_checkpoint(net_save_path, model, optimizer, epoch, logger)

        writer.close()
    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.output_dir), model, optimizer, epoch, logger)


if __name__ == '__main__':
    import utils
    from models.mobilenetv3_fpn import mobilenetv3_fpn

    #model = FPN_ResNet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.n)

    model = mobilenetv3_fpn(num_out=2, model_path='../MobileNetV3_large_x0_5.pth')

    # model = CRAFT(num_out=2, pretrained=True)

    criterion = Loss(OHEM_ratio=config.OHEM_ratio, reduction='mean')
    main(model, criterion)
