import cv2
import numpy as np
import os
import pyclipper
from shapely.geometry import Polygon

def fix_boxes(box, height=None, width=None):
    points = sorted(box, key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    new_box = [points[index_1], points[index_2], points[index_3], points[index_4]]

    if width != None:
        for i, pt in enumerate(new_box):
            pt[0] = 0 if pt[0] < 0 else pt[0]
            pt[1] = 0 if pt[1] < 0 else pt[1]
            pt[0] = width - 1 if pt[0] >= width else pt[0]
            pt[1] = height - 1 if pt[1] >= height else pt[1]
            new_box[i] = pt

    return new_box


def get_mini_boxes(box, height=None, width=None):
    if len(box) == 0:
        print('box empty')
        return None, None
    try:
        bounding_box = cv2.minAreaRect(box)
    except:
        print(box)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1].tolist(), points[index_2].tolist(), points[index_3].tolist(), points[index_4].tolist()]

    if width != None:
        for i, pt in enumerate(box):
            pt[0] = 0 if pt[0] < 0 else pt[0]
            pt[1] = 0 if pt[1] < 0 else pt[1]
            pt[0] = width - 1 if pt[0] >= width else pt[0]
            pt[1] = height - 1 if pt[1] >= height else pt[1]
            box[i] = pt

    return box, min(bounding_box[1])


def save_boxes(save_path, boxes):
    lines = []
    for i, bbox in enumerate(boxes):
        line = ''
        for box in bbox:
            line += "%d, %d, " % (int(box[0]), int(box[1]))

        line = line.rsplit(',', 1)[0]
        line += '\n'
        lines.append(line)

    with open(save_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def boxes_unclip(gt_path, img_path, save_path, unclip_val = 1):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    img_names = os.listdir(img_path)
    gt_names = [name.split('.')[0]+'.txt' for name in img_names]

    for j, name in enumerate(gt_names):
        if os.path.exists(os.path.join(save_path, name)):
            continue

        with open(os.path.join(gt_path, name), 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines()]
            boxes = []
            for line in lines:
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                if len(params) != 8:
                    print(name)
                    print(len(params))
                    print(params)
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, params))
                boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        new_boxes = []
        for i, box in enumerate(boxes):
            img = os.path.join(img_path, img_names[j])
            img = cv2.imread(img)
            height, width = img.shape[:2]

            new_box = fix_boxes(box, height, width)

            new_box = np.array(new_box)
            try:
                new_box = unclip(new_box, unclip_val)
            except:
                print(img_names[i])
                raise ValueError('123')
            try:
                new_box = new_box.reshape(-1, 1, 2)
                new_box, minval = get_mini_boxes(new_box, height, width)
            except:
                print(img_names[i])
                raise ValueError('123')

            if new_box == None:
                continue

            new_boxes.append(new_box)

        save_boxes(os.path.join(save_path, name), new_boxes)



if __name__ == '__main__':
    boxes_unclip('./gt', './img', './save', unclip_val=0.5)
    print('finished.')