import os
import cv2

path = './img'


def convert_name(img_path, gt_path=None, extention='.json'):
    img_names = os.listdir(img_path)

    for i, img in enumerate(img_names, start=1):
        extension = '.'+img.rsplit('.', 1)[-1]
        new_name = os.path.join(img_path, str(i)+extension)
        if not os.path.exists(new_name):
            os.rename(os.path.join(img_path, img), new_name)
        if gt_path != None:
            if not os.path.exists(os.path.join(gt_path, img.split('.')[0]+extention)):
                print(img.split('.')[0]+extention, ' is not exist. ---> ', str(i))
                continue
            os.rename(os.path.join(gt_path, img.split('.')[0]+extention), os.path.join(gt_path, str(i)+extention))


####################################################################

# for img_name in img_names:
#     img_name = os.path.join(path, img_name)
#     img = cv2.imread(img_name)    #不能有中文路径
#     h, w = img.shape[:2]
#     if max(h, w) > 2000:
#         scale = 2000 / max(h, w)
#         img = cv2.resize(img, None, fx=scale, fy=scale)
#         cv2.imwrite(img_name, img)


if __name__ == '__main__':
    convert_name('./img', './gt', '.json')
    print('finished.')
