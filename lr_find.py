# pip install torch-lr-finder
# https://github.com/davidtvs/pytorch-lr-finder#:~:text=PyTorch%20learning%20rate%20finder%20A%20PyTorch%20implementation%20of,provides%20valuable%20information%20about%20the%20optimal%20learning%20rate.
from torch_lr_finder import LRFinder

from models.fapn_resnet import FaPN_ResNet
from models.fapn_vgg16 import FaPN_VGG16_bn
from models.dla_seg import get_dlaseg_net
from models.loss import Loss

import config

from utils.radam import RAdam
from utils.ranger import Ranger
from utils.adabound import AdaBound, AdaBoundW
from utils.over9000 import Over9000
from utils.lamb import Lamb, log_lamb_rs
from utils.lamb_v3 import Lamb as Lamb_v3
from utils.la_lamb import La_Lamb, La_Lamb_v3

from dataset.CurveDataset import CurveDataset
from dataset.data_utils import DataLoaderX
from torchvision import transforms


model = get_dlaseg_net(34, heads={'seg_hm': 2})
criterion = Loss(OHEM_ratio=config.OHEM_ratio, reduction='mean')
optimizer = Ranger([{'params': model.parameters(), 'initial_lr': 1e-7}], lr=1e-7,
                   weight_decay=config.weight_decay)

train_data = CurveDataset(config.trainroot, data_shape=config.data_shape, dataset_type=config.dataset_type,
                          transform=transforms.ToTensor())
# train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
#                                num_workers=int(config.workers))
train_loader = DataLoaderX(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                           num_workers=int(config.workers), pin_memory=config.pin_memory)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph