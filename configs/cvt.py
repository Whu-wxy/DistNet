import os

# label前缀变成res_img_

path = '../test_result/result/'

img_paths = os.listdir(path)


for img_path in img_paths:
    img_name = os.path.basename(img_path).split('.')[0]
    new_name = 'res_img_' + img_name.split('_')[-1]+'.txt'
    os.rename(path+img_name+'.txt', path+new_name)

print('finish')
