
import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16_bn

if __name__ == '__main__':
    import time

    device = torch.device('cpu')
    model = vgg16_bn().to(device)
    model.eval()

    start = time.time()
    data = torch.randn(1, 3, 256, 256).to(device)
    output = model(data)
    print(time.time() - start)
    for u in output:
        print(u.shape)

    from utils.computation import print_model_parm_flops, print_model_parm_nums, show_summary

    print_model_parm_flops(model, data)
    print_model_parm_nums(model)
    # show_summary(model, input_shape=(3, 256, 256), save_path='E:/mysummery.xlsx')