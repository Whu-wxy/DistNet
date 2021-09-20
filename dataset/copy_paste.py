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

import pathlib
import os
from tqdm import tqdm
import glob

class CopyPaste(object):
    def __init__(self, objects_paste_ratio=0.2, limit_paste=True, iou = 0.2, scales = [0.5, 2], angle=[-45,45], **kwargs):
        self.ext_data_num = 1
        self.objects_paste_ratio = objects_paste_ratio  # 复制百分之多少的text
        self.limit_paste = limit_paste
        self.iou = iou
        self.scales = scales
        self.angle = angle

    def __call__(self, data):
        src_img = data['image']
        src_polys = data['polys']
        src_ignores = data['ignore_tags']

        indexs = [i for i in range(len(src_ignores)) if not src_ignores[i]]
        if len(indexs) == 0 or len(src_polys) == 0:
            return data

        select_num = max(1, min(int(self.objects_paste_ratio * len(src_polys)), 30))

        random.shuffle(indexs)
        select_idxs = indexs[:select_num]
        select_polys = [src_polys[idx] for idx in select_idxs]
        select_ignores = [src_ignores[idx] for idx in select_idxs]

        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_img).convert('RGBA')
        text_imgs = []
        new_polys = []
        for poly, tag in zip(select_polys, select_ignores):
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
            new_poly = poly.copy()
            new_poly[:, 0] -= xmin
            new_poly[:, 1] -= ymin
            new_polys.append(new_poly.tolist())
            text_img = text_img.crop((xmin, ymin, xmax, ymax))
            text_imgs.append(text_img)
            # text_img.show()
            # getEdge(text_img)

        for text_img, poly, tag in zip(text_imgs, new_polys, select_ignores):
            src_img, new_poly = self.paste_img(src_img, text_img, poly, src_polys)
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

    def paste_img(self, src_img, text_img, poly, src_polys):
        text_img = self.resize(text_img)

        src_w, src_h = src_img.size
        box_w, box_h = text_img.size

        angle = np.random.randint(self.angle[0], self.angle[1])  # 0, 360
        new_poly = rotate_bbox(text_img, poly, angle)   # box：四个点
        text_img = text_img.rotate(angle, expand=1)  # 旋转
        box_w, box_h = text_img.width, text_img.height
        if src_w - box_w < 0 or src_h - box_h < 0:
            print(src_w - box_w)
            print(src_h - box_h)
            print('after rotate failed')
            return src_img, None

        paste_x, paste_y = self.select_coord(src_polys, new_poly, src_w - box_w, src_h - box_h)# 平移确定位置,src_w - box_w防止超出图像
        if paste_x is None:
            print('select_coord failed')
            return src_img, None
        new_poly[:, 0] += paste_x
        new_poly[:, 1] += paste_y
        r, g, b, A = text_img.split()
        src_img.paste(text_img, (paste_x, paste_y), mask=A)

        return src_img, new_poly

    def resize(self, text_img):
        min = self.scales[0] * 10
        max = self.scales[-1] * 10
        rd_scale = np.random.randint(min, max, 1)
        rd_scale = rd_scale / 10

        return text_img.resize((rd_scale*text_img.size[0], rd_scale*text_img.size[1]), Image.BILINEAR)

    def select_coord(self, src_polys, new_poly, endx, endy):
        if self.limit_paste:
            for _ in range(50):
                randn_poly = new_poly.copy()
                paste_x = random.randint(0, endx)
                paste_y = random.randint(0, endy)
                randn_poly[:, 0] += paste_x
                randn_poly[:, 1] += paste_y

                bValid = True
                for poly in src_polys:
                    iou = get_intersection_over_union(poly, randn_poly.tolist())
                    # print('iou: ',iou)
                    if iou > self.iou:   # 和原多边形有交集
                        print('iou: ', iou)
                        bValid = False
                        break
                if bValid:
                    return paste_x, paste_y
            return None, None
        else:
            paste_x = random.randint(0, endx)
            paste_y = random.randint(0, endy)
            return paste_x, paste_y

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


def load_data_15(data_dir: str) -> list:
    data_list = []
    img_list = os.listdir(data_dir + '/img')
    for x in img_list:
        d = pathlib.Path(x)
        label_path = os.path.join(data_dir, 'gt', ('gt_' + str(d.stem) + '.txt'))
        bboxs, text = get_annotation_15(label_path)
        if len(bboxs) > 0:
            x_path = os.path.join(data_dir, 'img', x)
            img = cv2.imread(x_path)
            data_list.append({'image': img, "polys": bboxs, "ignore_tags": text})
        else:
            print('there is no suit bbox on {}'.format(label_path))
    return data_list

def get_annotation_15(label_path: str) -> tuple:
    boxes = []
    text_tags = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            if ',' in line:
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            else:
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(' ')
            try:
                if len(params) >= 8:
                    label = params[8]
                    if label == '*' or label == '###':   #在loss中用mask去掉
                        text_tags.append(True)  # True
                    else:
                        text_tags.append(False)  # False
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            except:
                print('load label failed on {}'.format(label_path))
    return boxes, text_tags

def load_data_curve(data_dir: str, dataset_type) -> list:
    data_list = []

    for x in glob.glob(data_dir + '/img/*.jpg', recursive=True):
        d = pathlib.Path(x)
        if dataset_type == 'ctw1500':
            label_path = os.path.join(data_dir, 'gt', (str(d.stem) + '.txt'))
        elif dataset_type == 'total':
            label_path = os.path.join(data_dir, 'gt', 'poly_gt_' + (str(d.stem) + '.txt'))
            # label_path = os.path.join(data_dir, 'gt', (str(d.stem) + '.txt'))
        else:
            raise Exception('数据集类型必须是ctw1500或total！')
        bboxs, text = get_annotation_curve(label_path, dataset_type)
        if len(bboxs) > 0:
            img = cv2.imread(x)
            data_list.append({'image': img, "polys": bboxs, "ignore_tags": text})
        else:
            print('there is no suit bbox on {}'.format(label_path))
    return data_list

def get_annotation_curve(label_path: str, dataset_type) -> tuple:
    boxes = []
    text_tags = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '').strip('\ufeff').strip('\xef\xbb\xbf')
            params = line.split(',')
            try:
                if dataset_type == 'ctw1500':
                    text_tags.append(False)
                    xmin, ymin, w, h = list(map(float, params[:4]))
                    box = []
                    x = 0
                    y = 0
                    for i, val in enumerate(params[4:]):
                        if i % 2 == 0:
                            x = xmin + int(val)
                        elif i % 2 == 1:
                            y = ymin + int(val)
                            box.append([x, y])
                    boxes.append(box)
                elif dataset_type == 'total':
                    box = []
                    x = 0
                    y = 0
                    if not params[-1].isdigit():
                        if line.endswith('#'):
                            text_tags.append(True)
                        else:
                            text_tags.append(False)
                        params.pop(-1)
                    else:
                        text_tags.append(False)
                    for i, val in enumerate(params):
                        if i % 2 == 0:
                            x = int(val)
                        elif i % 2 == 1:
                            y = int(val)
                            box.append([x, y])
                    boxes.append(box)
            except:
                print('load label failed on {}'.format(label_path))

    if dataset_type == 'total':   # padding
        pt_count = 0
        for box in boxes:
            if len(box) > pt_count:
                pt_count = len(box)
        for i, box in enumerate(boxes):
            if len(box) < pt_count:
                box2 = box
                for j in range(len(box), pt_count):
                    box2.append(box[-1])
                boxes[i] = box2

    return boxes, text_tags

def cvt_15(data_dir, save_dir):
    cur_num = 0
    img_dir = os.path.join(save_dir, 'img')
    gt_dir = os.path.join(save_dir, 'gt')
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        os.makedirs(img_dir)
        os.makedirs(gt_dir)
    else:
        cur_num = len(os.listdir(img_dir))

    data_list = load_data_15(data_dir)
    pbar = tqdm(total=len(data_list))
    cp = CopyPaste(0.5, True, 0.05, scales = [1, 2])
    for i, data in enumerate(data_list, cur_num+1):
        pbar.update(1)
        data = cp(data)
        cv2.imwrite(os.path.join(img_dir, str(i)+'.jpg'), data['image'])
        # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
        # cv2.imshow('res', data['image'])
        # cv2.waitKey()


def cvt_curve(data_dir, save_dir, dataset_type):
    cur_num = 0
    img_dir = os.path.join(save_dir, 'img')
    gt_dir = os.path.join(save_dir, 'gt')
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        os.makedirs(img_dir)
        os.makedirs(gt_dir)
    else:
        cur_num = len(os.listdir(img_dir))

    data_list = load_data_curve(data_dir, dataset_type)
    pbar = tqdm(total=len(data_list))
    cp = CopyPaste(0.5, True, 0.05, scales = [1, 2])
    for i, data in enumerate(data_list, cur_num+1):
        pbar.update(1)
        data = cp(data)
        cv2.imwrite(os.path.join(img_dir, str(i)+'.jpg'), data['image'])
        # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
        # cv2.imshow('res', data['image'])
        # cv2.waitKey()

if __name__ == '__main__':

    # # F:\\zzxs\\Experiments\\dl-data\\TotalText\\test\\img\\img1.jpg
    # # F:\\zzxs\\Experiments\\dl-data\\ICDAR\\ICDAR2013\\Challenge2_Test_Task12_Images\\img_18.jpg
    # img = cv2.imread('F:\\zzxs\\Experiments\\dl-data\\TotalText\\test\\img\\img1.jpg')
    # # [[[206,633],[251,811],[386,931],[542,946],[620,926],[646,976],[550,1009],[358,989],[189,845],[140,629]]]
    # # [[[340,295], [482,295], [482,338], [340, 338]]]
    # polys = [[[206,633],[251,811],[386,931],[542,946],[620,926],[646,976],[550,1009],[358,989],[189,845],[140,629]]]
    # tags = [False]
    #
    # cp = CopyPaste(0.2, True, 0.2)
    # data = {'image': img, "polys": polys, "ignore_tags": tags}
    # data = cp(data)
    # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    # cv2.imshow('res',data['image'])
    # cv2.waitKey()

    # i = get_intersection([[340,295], [482,295], [482,338], [340, 338]], [[345,295], [487,295], [487,338], [347, 338]])
    # print('i:', i)

    # cvt_15('F:\\zzxs\\Experiments\\dl-data\\ICDAR\\ICDAR2015\\sample_IC15\\train',
    #        'F:\\zzxs\\Experiments\\dl-data\\ICDAR\\ICDAR2015\\sample_IC15\\train_res')

    cvt_curve('F:\\zzxs\\Experiments\\dl-data\\TotalText\\sample',
              'F:\\zzxs\\Experiments\\dl-data\\TotalText\\res', 'total')
