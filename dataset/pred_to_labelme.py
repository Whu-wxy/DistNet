import os
import json
import cv2

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



if __name__ == '__main__':
    text2json('F:\\train\\gt', 'F:\\train\\gt_json', 'F:\\train\\img')
    print('finished.')



