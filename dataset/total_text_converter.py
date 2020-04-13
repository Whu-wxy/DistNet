# 正则表达式库
import re
import cv2
import os
import numpy as np
from tqdm import tqdm

# Total-Text To IC15

# F:\zzxs\Experiments\dl-data\TotalText\Groundtruth\Text\Legacy
# F:\zzxs\Experiments\dl-data\TotalText\Groundtruth\Text\Legacy\\txt_format\Test
# F:\zzxs\Experiments\dl-data\TotalText\Groundtruth\Text\\txt_format\Train
# F:\zzxs\Experiments\dl-data\CTW\ctw1500_e2e_annos\ctw1500_e2e_train
# F:\zzxs\Experiments\dl-data\ICDAR\ICDAR2015\\train\gt
root_path = 'F:\zzxs\Experiments\dl-data\TotalText\Groundtruth\Text\\txt_format\Train'
_indexes = os.listdir(root_path)


withTranscription = True

def count_donotcare():
    all_count = 0
    donotcare_count = 0
    for index in tqdm(_indexes, desc='count_ctw_donotcare'):
        if os.path.splitext(index)[1] != '.txt':
            continue
        anno_file = os.path.join(root_path, index)
        with open(anno_file, 'r+', encoding='utf-8') as f:
            # lines是每个文件中包含的内容
            lines = [line for line in f.readlines()]
            for line in lines:
                line = line.strip()
                if len(line) < 4:
                    continue
                if line[-3] == '#':
                    donotcare_count += 1
                all_count += 1

    print(donotcare_count)
    print(all_count)
    print('rate: ', float(donotcare_count)/all_count)

def cvt_total_text():
    invalid_count = 0
    all_count = 0
    for index in tqdm(_indexes, desc='convert labels'):
        if os.path.splitext(index)[1] != '.txt':
            continue
        anno_file = os.path.join(root_path, index)

        with open(anno_file, 'r+') as f:
            # lines是每个文件中包含的内容
            lines = [line for line in f.readlines() if line.strip()]
            single_list = []
            all_list = []
            try:
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    while not line.endswith(']'):
                        print('concat in ', index)
                        i = i + 1
                        line = line + ' ' + lines[i].strip()
                    i += 1

                    if line[-3] == '#':
                        invalid_count += 1
                    all_count += 1

                    parts = line.split(',')
                    xy_list = []
                    for a, part in enumerate(parts):
                        if a > 1:
                            break
                        piece = part.strip().split(',')
                        numberlist = re.findall(r'\d+', piece[0])
                        xy_list.extend(numberlist)

                    length = len(xy_list)
                    n = int(length / 2)
                    x_list = xy_list[:n]
                    y_list = xy_list[n:]
                    single_list = [None] * (len(x_list) + len(y_list))
                    single_list[::2] = x_list
                    single_list[1::2] = y_list

                    if withTranscription:
                        parts = line.split('\'')
                        transcription = parts[-2]
                        single_list.append(transcription)
                    all_list.append(single_list)

            except Exception as e:
                print('error: ', index)

        with open(anno_file, 'w') as w:
            for all_list_piece in all_list:
                w.write(','.join(all_list_piece))
                w.write('\n')
    print('### count: ', invalid_count)
    print('All count: ', all_count)
    print('rate: ', float(invalid_count)/all_count)

if __name__ == '__main__':
    cvt_total_text()