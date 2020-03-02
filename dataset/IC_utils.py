import os
import glob

def convert2PointTo4Point(in_path):
    if not os.path.exists(os.path.join(in_path, 'gt2')):
        os.mkdir(os.path.join(in_path, 'gt2'))
    for x in glob.glob(in_path + '/gt/*.txt'):
        out_name = in_path + '/gt2/'+ os.path.basename(x)

        with open(x, encoding='utf-8', mode='r') as f:
            with open(out_name, encoding='utf-8', mode='w') as f2:
                for line in f.readlines():
                    if ',' in line and line.count(',') > 3:
                        params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                    else:
                        params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(' ')
                    label = params[-1]
                    if label == '*' or label == '###':
                        continue
                    x1 = x4 = int(params[0])
                    y1 = y2 = int(params[1])
                    x2 = x3 = int(params[2])
                    y3 = y4 = int(params[3])

                    data = [x1, y1, x2, y2, x3, y3, x4, y4, label]
                    data = [str(num) for num in data]
                    f2.write(','.join(data) + '\n')

convert2PointTo4Point('../../data/IC13/test/')
print('finished')