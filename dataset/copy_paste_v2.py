import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import cv2

# 前面图片中的text放到后面的图片中
class CopyPaste_v2(object):
    def __init__(self, objects_paste_ratio=0.2, limit_paste=True, iou = 0.2, scales = [0.5, 2], angle=[-45,45], **kwargs):
        super().__init__(objects_paste_ratio, limit_paste, iou, scales, angle, kwargs)



    def __call__(self, data):
        pass