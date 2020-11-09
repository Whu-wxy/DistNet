import os
import cv2

def convert(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    sources = os.listdir(src_path)
    for src in sources:
        path = os.path.join(src_path, src)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        with open(os.path.join(dst_path, src.split('.')[0] + '.txt'), 'w') as f:
            pts = [0, 0, 0, h-1, w-1, h-1, h-1, 0]
            f.write(', '.join(('%s' %pt for pt in pts)))

if __name__ == '__main__':
    convert('./singlechar100', './gt')
    print('finished.')
