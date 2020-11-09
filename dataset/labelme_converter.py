import os
import json
import cv2
import numpy as np

#txt to labelme format
def text2json(text_path, json_path, img_path):
    if not os.path.exists(text_path):
        print('text_path is not exist.')
        return
    if not os.path.exists(json_path):
        os.mkdir(json_path)

    label_names = os.listdir(text_path)
    for name in label_names:
        json_data = {}
        json_data["version"] = "4.5.6"
        json_data["flags"] = {}
        shapes = []
        with open(os.path.join(text_path, name), 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines()]
            for line in lines:
                shape = {}
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, params[:8]))
                shape["label"] = "text"
                shape["points"] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                shape["group_id"] = None,
                shape["shape_type"] = "polygon"
                shape["flags"] = {}
                shapes.append(shape)

        json_data["shapes"] = shapes
        json_data["imagePath"] = "..\\img\\" + name.rsplit('.')[0] + ".jpg"
        json_data["imageData"] = None
        img = cv2.imread(os.path.join(img_path, name.rsplit('.')[0] + ".jpg"))
        h, w = img.shape[:2]
        json_data["imageHeight"] = h
        json_data["imageWidth"] = w

        json_name = os.path.join(json_path, name.rsplit('.')[0] + '.json')
        with open(json_name, 'w') as f:
            f.write(json.dumps(json_data))

#labelme format to txt
def json2text(json_path, text_path):
    if not os.path.exists(json_path):
        print('json_path is not exist.')
        return
    if not os.path.exists(text_path):
        os.mkdir(text_path)
    json_names = os.listdir(json_path)

    for name in json_names:
        json_data = {}
        with open(os.path.join(json_path, name), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        shapes = json_data["shapes"]

        text_name = os.path.join(text_path, name.rsplit('.')[0] + '.txt')
        with open(text_name, 'w') as f:

            for shape in shapes:
                # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = shape["points"]
                # f.write(str(x1) + ', ' + str(y1) + ', ' + str(x2) + ', ' + str(y2) + ', ' + str(x3) + ', ' + str(y3) + ', ' + str(x4) + ', ' + str(y4) + '\n')
                shape = np.array(shape["points"])
                ###
                shape = np.trunc(shape)
                shape = shape.astype(int)
                ###
                shape = shape.tolist()
                length = len(shape)
                if length == 1:
                    continue
                if length == 2:
                    shape = [shape[0], [shape[1][0], shape[0][1]], shape[1], [shape[0][0], shape[1][1]] ]
                # clear[]
                s = str(shape).replace('[', '').replace(']', '') + '\n'
                f.write(s)

def check_img_and_gt(img_path, gt_path):
    img_names = [int(img.split('.')[0]) for img in os.listdir(img_path)]
    gt_names = [int(gt.split('.')[0]) for gt in os.listdir(gt_path)]

    for img in img_names:
        if img not in gt_names:
            print(img, '\n')


def find_multi_point_gt(gt_path):
    gt_names = os.listdir(gt_path)
    for name in gt_names:
        with open(os.path.join(gt_path, name), 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines()]
            for line in lines:
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                if len(params) > 8:
                    print("Contain multi_point: ", name)
                    break

if __name__ == '__main__':
    # text2json('F:\\train\\gt', 'F:\\train\\gt_json', 'F:\\train\\img')
    json2text('./gt_json', './gt')
    # check_img_and_gt('./img', './gt_txt')
    # find_multi_point_gt('./gt')
    print('finished.')



