from detectron2.modeling import build_model
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import torch
from detectron2.modeling.backbone.build import build_backbone
from detectron2.modeling.backbone.resnet import build_resnet_backbone

def load_dcn_resnet(pretrained = True, config_file = '/root/code/PSENet/models/dcn_resnet.yaml'):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    model = build_backbone(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    return model


if __name__ == '__main__':
    pretrained_model = load_dcn_resnet(True, '/root/code/PSENet/models/dcn_resnet.yaml')
    device = torch.device("cuda:0")
    images = torch.rand(1, 3, 640, 640)  # .cuda(0)  224, 224
    pretrained_model = pretrained_model.to(device)
    images = images.to(device)
    out = pretrained_model(images)
    print(out['res5'].shape)
    # for i in out:
    #     print(i.size())


    from models.resnet import resnet50
    from utils.utils import load_part_checkpoint

    net = resnet50(False)
    # load_part_checkpoint('/home/ubuntu/MyFiles/PublicData/model/pytorch/resnet50/resnet50-19c8e357.pth', net, part_id_list=[(0, -2)])
    DetectionCheckpointer(net).load('/home/ubuntu/MyFiles/PublicData/model/pytorch/resnet50/resnet50-19c8e357.pth')
    net = net.to(device)
    y = net(images)
    print(y[-1].shape)
