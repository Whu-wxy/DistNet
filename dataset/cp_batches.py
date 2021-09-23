from dataset.copy_paste_v2 import CopyPaste_v2
from dataset.copy_paste import CopyPaste
from tqdm import tqdm
import glob
import pathlib
import os
import cv2

def load_data_15(data_dir: str) -> list:
    data_list = []
    img_list = os.listdir(data_dir + '/img')
    for x in img_list:
        d = pathlib.Path(x)
        label_path = os.path.join(data_dir, 'gt', ('gt_' + str(d.stem) + '.txt'))
        bboxs, text = get_annotation_15(label_path)
        if len(bboxs) > 0:
            x_path = os.path.join(data_dir, 'img', x)
            data_list.append({'image': x_path, "polys": bboxs, "ignore_tags": text})
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
            data_list.append({'image': x, "polys": bboxs, "ignore_tags": text})
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
                    xmin, ymin, w, h = list(map(int, params[:4]))
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
        result = cp(data)
        # img = result['image']
        try:
            cv2.imwrite(os.path.join(img_dir, str(i)+'.jpg'), result['image'])
        except Exception as r:
            cv2.namedWindow('res', cv2.WINDOW_NORMAL)
            cv2.imshow('res', result['image'])
            cv2.waitKey()
            print('error： ', r)
        with open(os.path.join(gt_dir, 'gt_' + str(i)+'.txt'), 'w') as f:
            ignore_tags = result['ignore_tags']
            for j, box in enumerate(result['polys']) :
                box_str = ''
                for pt in box:
                    box_str += str(int(pt[0])) + ', ' + str(int(pt[1])) + ', '
                if ignore_tags[j]:
                    box_str += '###\n'
                else:
                    box_str += 'TEXT\n'
                f.write(box_str)
        # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
        # cv2.imshow('res', result['image'])
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
    cp = CopyPaste_v2(0.5, True, 0.05, scales = [1, 2])
    for i, data in enumerate(data_list, cur_num+1):
        pbar.update(1)
        result = cp(data)
        cv2.imwrite(os.path.join(img_dir, str(i)+'.jpg'), result['image'])
        gt_file = os.path.join(gt_dir, str(i) + '.txt')
        if dataset_type == 'total':
            gt_file = os.path.join(gt_dir, 'poly_gt_' + str(i) + '.txt')
        with open(gt_file, 'w') as f:
            ignore_tags = result['ignore_tags']
            for j, box in enumerate(result['polys']):
                box_str = ''
                if dataset_type == 'ctw1500':
                    box_str = '0, 0, 0, 0, '
                for pt in box:
                    box_str += str(pt[0]) + ', ' + str(pt[1]) + ', '
                if dataset_type == 'total':
                    if ignore_tags[j]:
                        box_str += '###\n'
                    else:
                        box_str += 'TEXT\n'
                else:
                    box_str = box_str[:-2] + '\n'
                f.write(box_str)
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

    cvt_curve('F:\\zzxs\\Experiments\\dl-data\\TotalText\\sample2',
              'F:\\zzxs\\Experiments\\dl-data\\TotalText\\res2', 'total')

    # cvt_curve('F:\zzxs\Experiments\dl-data\CTW\ctw1500\sample',
    #           'F:\zzxs\Experiments\dl-data\CTW\ctw1500\\res', 'ctw1500')