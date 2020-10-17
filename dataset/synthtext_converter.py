# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         synthText2TotalText
# Description:
# Author:       zx
# Date:         2020/9/18
# -------------------------------------------------------------------------------

import scipy.io as sio
import os
from tqdm import tqdm

dataFile = './gt.mat'
saveFile = "./SynthText"
train_images_gts = saveFile + "/gt"
train_images_list_txt = "train_list.txt"
if not os.path.exists(saveFile):
    os.makedirs(saveFile)
if not os.path.exists(train_images_gts):
    os.makedirs(train_images_gts)


def convert2txt():
    fh_r = open(os.path.join(saveFile, train_images_list_txt), 'w', encoding='utf-8')
    data = sio.loadmat(dataFile)
    for i in tqdm(range(len(data["imnames"][0]))):
        # write train_list.txt

        #data["imnames"][0][i][0] ----> 8/ballet_106_107.jpg
        gt_name = "{}.txt".format(data["imnames"][0][i][0].split("/")[1].split(".")[0])   # ballet_106_107.txt
        fh_r.write("{}, {}".format(data["imnames"][0][i][0], gt_name) + '\n')
        # write train_gts
        fh_gt = open(os.path.join(train_images_gts, gt_name), 'w', encoding='utf-8')
        # get word list
        rec = data['wordBB'][0][i]
        txt_str = ""
        for words in data["txt"][0][i]:
            txt_str += " " + " ".join([w.strip() for w in words.split("\n")])
        txt_str = txt_str.strip().split(" ")
        # # get word list
        # print(data["txt"][0][i])
        #print(txt_str)
        #print(len(txt_str), len(rec[0][0]))

        try:
            if len(rec.shape) == 3:
                for j in range(len(rec[0][0])):
                    x1 = int(rec[0][0][j])
                    y1 = int(rec[1][0][j])
                    x2 = int(rec[0][1][j])
                    y2 = int(rec[1][1][j])
                    x3 = int(rec[0][2][j])
                    y3 = int(rec[1][2][j])
                    x4 = int(rec[0][3][j])
                    y4 = int(rec[1][3][j])
                    fh_gt.write(str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2)
                                + "," + str(x3) + "," + str(y3) + "," + str(x4) + "," + str(y4) + "," + txt_str[j] + '\n')
            elif len(rec.shape) == 2:
                x1 = int(rec[0][0])
                y1 = int(rec[1][0])
                x2 = int(rec[0][1])
                y2 = int(rec[1][1])
                x3 = int(rec[0][2])
                y3 = int(rec[1][2])
                x4 = int(rec[0][3])
                y4 = int(rec[1][3])
                fh_gt.write(str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2)
                            + "," + str(x3) + "," + str(y3) + "," + str(x4) + "," + str(y4) + "," + txt_str[0] + '\n')

        except:
            raise ValueError(rec[0][0])
        fh_gt.close()
   
    fh_r.close()


if __name__ == '__main__':
    convert2txt()
