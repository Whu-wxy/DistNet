import os
import cv2

path = 'F:\\test'

img_names = os.listdir(path)
#
# for i, img in enumerate(img_names, start=1):
#     extension = '.'+img.rsplit('.', 1)[-1]
#     new_name = os.path.join(path, str(i)+extension)
#     if not os.path.exists(os.path.join(path, new_name)):
#         os.rename(os.path.join(path, img), new_name)


####################################################################

for img_name in img_names:
    img_name = os.path.join(path, img_name)
    img = cv2.imread(img_name)    #不能有中文路径
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.imwrite(img_name, img)
