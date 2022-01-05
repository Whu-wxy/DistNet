import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import cv2
from dataset.copy_paste import CopyPaste
import random
import os

# 前面图片中的text放到后面的图片中
class CopyPaste_v2(CopyPaste):
    def __init__(self, objects_paste_ratio=0.2, limit_paste=True, iou = 0.2, scales = [0.5, 2],
                 angle=[-45,45], use_shape_adaptor=False, colorjit=True, **kwargs):
        super().__init__(objects_paste_ratio, limit_paste, iou, scales, angle, use_shape_adaptor, colorjit)

        self.max_text_region_ratio = 0.4    # 文字与图片面积的比例0-0.4---->从buffer中取出的比例: 0.4-0
        self.buffer_size = 20
        self.text_img_buffer = []
        self.refresh_ratio = 0.2
        self.refresh_buffer_img_short_side_th = 500

    def __call__(self, data):
        src_img = data['image']
        src_polys = data['polys']
        src_ignores = data['ignore_tags']
        if isinstance(src_img, str):
            src_img = cv2.imread(src_img)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_img).convert('RGBA')

        text_ratio = self.cal_text_ratio(src_img, np.array(src_polys))
        # 文本面积比例比较大，不添加
        select_num = int((self.max_text_region_ratio - text_ratio)*self.buffer_size) if text_ratio<self.max_text_region_ratio else 0

        # 需要填满buffer
        if len(self.text_img_buffer) < self.buffer_size:
            fill_num = self.buffer_size - len(self.text_img_buffer)

            indexs = [i for i in range(len(src_ignores)) if not src_ignores[i]]
            if len(indexs) != 0 and len(src_polys) != 0:
                # 随机取出self.buffer_size个文本块
                random.shuffle(indexs)
                select_idxs = indexs[:fill_num]
                select_polys = [src_polys[idx] for idx in select_idxs]
                text_imgs, polys = self.get_text_imgs(src_img, select_polys)
                for i, p in zip(text_imgs, polys):
                    # i.show()
                    self.text_img_buffer.append({'image': i, 'polys': p})

        if len(self.text_img_buffer) > 0:
            # paste
            indexs = [i for i in range(len(src_ignores)) if not src_ignores[i]]
            for buffer in self.text_img_buffer[:select_num]:
                src_img, new_poly = self.paste_img(src_img, buffer['image'], buffer['polys'], src_polys, src_ignores)
                if new_poly is not None:
                    src_polys.append(new_poly.tolist())
                    src_ignores.append(False)

            # refresh buffer
            if len(indexs) != 0 and len(src_polys) != 0 and min(src_img.size) > self.refresh_buffer_img_short_side_th \
                    and len(self.text_img_buffer) == self.buffer_size:
                # 随机取出self.buffer_size个文本块
                random.shuffle(indexs)
                random.shuffle(self.text_img_buffer)
                refresh_count = min(len(indexs), int(self.refresh_ratio*len(self.text_img_buffer)))
                select_idxs = indexs[:refresh_count]
                select_polys = [src_polys[idx] for idx in select_idxs]
                text_imgs, polys = self.get_text_imgs(src_img, select_polys)
                for j, (i, p) in enumerate(zip(text_imgs, polys)):
                    self.text_img_buffer[j] = {'image': i, 'polys': p}

        src_img = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
        h, w = src_img.shape[:2]
        src_polys = pad_polys(src_polys)
        src_polys = np.array(src_polys)
        src_polys[:, :, 0] = np.clip(src_polys[:, :, 0], 0, w)
        src_polys[:, :, 1] = np.clip(src_polys[:, :, 1], 0, h)
        data['image'] = src_img
        data['polys'] = src_polys
        data['ignore_tags'] = np.array(src_ignores)
        return data

    def cal_text_ratio(self, img, polys):
        text_area = 0
        img_area = img.size[1] * img.size[0]
        for poly in polys:
            xmin, ymin, xmax, ymax = poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()
            text_area += (xmax - xmin) * (ymax - ymin)

        return text_area / img_area*1.0

    def get_text_imgs(self, src_img, polys):
        text_imgs = []
        new_polys = []
        # text_img_save_path = 'F:\\zzxs\\Experiments\\dl-data\\TotalText\\res\\blocks'
        # num = len(os.listdir(text_img_save_path))
        for poly in polys:
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
            if (xmax - xmin) * (ymax - ymin) > src_img_rgba.shape[1] * src_img_rgba.shape[0] * self.filt_large_text:
                # print('filt large text')
                continue
            new_poly = poly.copy()
            new_poly[:, 0] -= xmin
            new_poly[:, 1] -= ymin
            new_polys.append(new_poly.tolist())
            text_img = text_img.crop((xmin, ymin, xmax, ymax))
            # text_img.save(os.path.join(text_img_save_path, str(num)+'.png'))
            text_imgs.append(text_img)

        return text_imgs, new_polys

def pad_polys(boxes):
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
    return boxes