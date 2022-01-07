# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import math
import albumentations as A

class CopyPaste(object):
    def __init__(self, objects_paste_ratio=0.2, limit_paste=True, iou = 0.2, scales = [0.5, 2],
                 angle=[-45,45], use_shape_adaptor=False, colorjit=True, **kwargs):
        self.ext_data_num = 1
        self.objects_paste_ratio = objects_paste_ratio  # 复制百分之多少的text
        self.limit_paste = limit_paste
        self.iou = iou
        self.scales = scales
        self.angle = angle
        self.filt_large_text = 0.1 #大于图片面积百分比的文本不复制
        self.use_shape_adaptor = use_shape_adaptor
        self.colorjit = colorjit
        self.colorjitter = A.ColorJitter(always_apply=True, p=1)

    def __call__(self, data):
        src_img = data['image']
        src_polys = data['polys']
        src_ignores = data['ignore_tags']
        src_img = cv2.imread(src_img)

        indexs = [i for i in range(len(src_ignores)) if not src_ignores[i]]
        if len(indexs) == 0 or len(src_polys) == 0:
            data['image'] = src_img
            return data

        select_num = max(1, min(int(self.objects_paste_ratio * len(src_polys)), 30))
        if len(src_polys) <= 5:   # 少于5个的，直接复制，多于5个的，复制一定比例，而且不超过30个
            select_num = len(src_polys)

        random.shuffle(indexs)
        select_idxs = indexs[:select_num]
        select_polys = [src_polys[idx] for idx in select_idxs]
        select_ignores = [src_ignores[idx] for idx in select_idxs]

        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_img).convert('RGBA')
        text_imgs = []
        new_polys = []
        for poly, tag in zip(select_polys, select_ignores):
            # poly roi
            src_img_rgba = np.asarray(src_img)
            maskIm = Image.new('L', (src_img_rgba.shape[1], src_img_rgba.shape[0]), 0)
            poly_t = [tuple(pt) for pt in poly]
            ImageDraw.Draw(maskIm).polygon(poly_t, outline=1, fill=1)
            mask = np.array(maskIm)
            text_img = np.empty(src_img_rgba.shape, dtype='uint8')
            text_img[:, :, :3] = src_img_rgba[:, :, :3]
            text_img[:, :, 3] = mask * 255
            text_img = Image.fromarray(text_img, "RGBA")
            poly = np.array(poly)
            xmin, ymin, xmax, ymax = poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()
            if (xmax-xmin)*(ymax-ymin) > src_img_rgba.shape[1]*src_img_rgba.shape[0]*self.filt_large_text:
                # print('filt large text')
                continue
            new_poly = poly.copy()
            new_poly[:, 0] -= xmin
            new_poly[:, 1] -= ymin
            new_polys.append(new_poly.tolist())
            text_img = text_img.crop((xmin, ymin, xmax, ymax))
            text_imgs.append(text_img)
            # text_img.show()
            # getEdge(text_img)

        for text_img, poly, tag in zip(text_imgs, new_polys, select_ignores):
            src_img, new_poly = self.paste_img(src_img, text_img, poly, src_polys, src_ignores)
            if new_poly is not None:
                src_polys.append(new_poly.tolist())
                src_ignores.append(tag)
        src_img = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
        h, w = src_img.shape[:2]
        src_polys = np.array(src_polys)
        src_polys[:, :, 0] = np.clip(src_polys[:, :, 0], 0, w)
        src_polys[:, :, 1] = np.clip(src_polys[:, :, 1], 0, h)
        data['image'] = src_img
        data['polys'] = src_polys
        data['ignore_tags'] = np.array(src_ignores)
        return data

    def paste_img(self, src_img, text_img, poly, src_polys, select_ignores):
        if self.colorjit:
            text_img = cv2.cvtColor(np.array(text_img), cv2.COLOR_RGB2BGR)
            text_img = self.colorjitter(image=text_img)
            text_img = Image.fromarray(text_img['image']).convert('RGBA')

        if self.use_shape_adaptor:
            text_img, poly = self.shape_adaptor(text_img, np.array(poly, dtype=np.float64), np.array(src_polys), select_ignores)

        text_img, poly = self.resize(text_img, np.array(poly, dtype=np.float64))

        src_w, src_h = src_img.size
        box_w, box_h = text_img.size

        angle = np.random.randint(self.angle[0], self.angle[1])  # 0, 360
        new_poly = rotate_bbox(text_img, poly, angle)   # box：四个点
        text_img = text_img.rotate(angle, expand=1)  # 旋转
        box_w, box_h = text_img.width, text_img.height
        if src_w - box_w < 0 or src_h - box_h < 0:
            # print(src_w - box_w)
            # print(src_h - box_h)
            # print('after rotate failed')
            return src_img, None

        paste_x, paste_y = self.select_coord(src_polys, new_poly, src_w - box_w, src_h - box_h)# 平移确定位置,src_w - box_w防止超出图像
        if paste_x is None:
            # print('select_coord failed')
            return src_img, None
        new_poly[:, 0] += paste_x
        new_poly[:, 1] += paste_y
        r, g, b, A = text_img.split()
        src_img.paste(text_img, (paste_x, paste_y), mask=A)

        return src_img, new_poly

    def resize(self, text_img, poly):
        min = self.scales[0] * 10
        max = self.scales[-1] * 10
        rd_scale = np.random.randint(min, max, 1)
        rd_scale = rd_scale / 10
        return self.resize_(text_img, poly, rd_scale)

    def resize_(self, text_img, poly, rd_scale):
        text_img = text_img.resize((int(rd_scale * text_img.size[0]), int(rd_scale * text_img.size[1])), Image.BILINEAR)
        poly *= rd_scale
        return text_img, poly.tolist()   # .astype(np.uint8)

    def select_coord(self, src_polys, new_poly, endx, endy):
        if self.limit_paste:
            for _ in range(10):
                randn_poly = new_poly.copy()
                paste_x = random.randint(0, endx)
                paste_y = random.randint(0, endy)
                randn_poly[:, 0] += paste_x
                randn_poly[:, 1] += paste_y

                bValid = True
                for poly in src_polys:
                    try:
                        iou = get_intersection_over_union(poly, randn_poly.tolist())
                    except Exception as e:
                        # print('iou cal Exception.')
                        continue
                    # print('iou: ',iou)
                    if iou > self.iou:   # 和原多边形有交集
                        # print('iou: ', iou)
                        bValid = False
                        break
                if bValid:
                    return paste_x, paste_y
            return None, None
        else:
            paste_x = random.randint(0, endx)
            paste_y = random.randint(0, endy)
            return paste_x, paste_y

    def shape_adaptor(self, text_img, text_poly, polys, select_ignores):
        # 长边中位数
        long_sides = []
        for poly, tag in zip(polys, select_ignores):
            if tag:
                continue
            poly = np.array(poly)
            xmin, ymin, xmax, ymax = poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()
            # long_side = max((xmax - xmin), (ymax - ymin))
            long_side = (xmax - xmin) * (ymax - ymin)
            long_sides.append(long_side)
        long_side_media = np.median(long_sides)
        scale = math.sqrt(long_side_media*1.0 / (text_img.size[0]*text_img.size[1]))  #max(text_img.size)
        # print('adaptor scale: ', scale)
        text_img, text_poly = self.resize_(text_img, text_poly, scale)
        return text_img, text_poly

def get_union(pD, pG):
    return Polygon(pD).union(Polygon(pG)).area

def get_intersection_over_union(pD, pG):
    return get_intersection(pD, pG) / get_union(pD, pG)

def get_intersection(pD, pG):
    return Polygon(pD).intersection(Polygon(pG)).area

def rotate_bbox(img, poly, angle, scale=1):
    """
    from https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/augment.py
    Args:
        img: np.ndarray
        poly: np.ndarray M*N*2
        angle: int
        scale: int
    Returns:
    """
    w, h = img.size

    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    # ---------------------- rotate box ----------------------
    rot_text_polys = list()
    for pt in poly:
        point = np.dot(rot_mat, np.array([pt[0], pt[1], 1]))
        rot_text_polys.append(point)
    return np.array(rot_text_polys, np.int32)

def getEdge(text_img):
    from PIL import ImageFilter
    r, g, b, A = text_img.split()
    maskIm = Image.new('L', (text_img.size[0], text_img.size[1]), 0)
    text_img = text_img.convert("L")
    text_img = text_img.filter(ImageFilter.FIND_EDGES)
    maskIm.paste(text_img, (0, 0), mask=A)
    maskIm.show()
    return maskIm


