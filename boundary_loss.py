import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res

#batch整体处理
def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


#batch内一个一个处理
def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]   #这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    # b1wh, b1wh
    def __call__(self, probs: Tensor, dist_maps: Tensor, mask: Tensor=None) -> Tensor:
        # assert simplex(probs)
        # assert not one_hot(dist_maps)

        # pc = probs[:, self.idc, ...].type(torch.float32)
        # dc = dist_maps[:, self.idc, ...].type(torch.float32)

        if mask != None:
            probs = probs * mask
            dist_maps = dist_maps * mask

        multipled = einsum("bcwh,bcwh->bcwh", probs, dist_maps)
        loss = multipled.mean()

        return loss

def boundary_loss(probs: Tensor, dist_maps: Tensor, mask: Tensor=None) -> Tensor:
    if mask != None:
        probs = probs * mask
        dist_maps = dist_maps * mask

    multipled = einsum("bcwh,bcwh->bcwh", probs, dist_maps)
    loss = multipled.mean()

    return loss


if __name__ == "__main__":
    #label
    # CWH
    # data = torch.tensor([[[0, 1, 1, 1, 1, 1, 0],
    #                       [0, 1, 1, 1, 1, 0, 0],
    #                       [0, 1, 1, 1, 1, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0]]])

    data = torch.tensor([[[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]])


    # BCWH
    data2 = class2one_hot(data, 2)
    print(data2.shape)
    # CWH
    data2 = data2[0].numpy()
    print(data2.shape)
    data3 = one_hot2dist(data2)   #cwh

    data3 = data3 + 2
    print(data3)
    print("data3.shape:", data3.shape)

    # predict class
    # logits = torch.tensor([[[0, 1, 1, 1, 0, 0, 0],
    #                          [0, 1, 1, 1, 1, 1, 0],
    #                          [0, 1, 1, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 0, 0]]])

    probs = torch.tensor([[[[0., 1., 1., 1., 0.9, 0.9, 0.],
                          [0., 1, 1., 0.9, 0.9, 0, 0.],
                          [0., 1., 1., 1., 0.9, 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0.]]]])


    # logits = class2one_hot(logits, 2)

    # # logits = logits[0].numpy()
    # # logits = one_hot2dist(logits)
                
    Loss = SurfaceLoss()
    # # logits = torch.tensor(logits)
    data3 = data3[1, ...]
    dist_map = torch.tensor(data3).type(torch.float32).unsqueeze(0).unsqueeze(0)

    res = Loss(probs, dist_map)
    print('loss:', res)


