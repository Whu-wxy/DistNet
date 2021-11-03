from dataset.cp_batches import cvt_curve, cvt_15

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

########################################## for test
    # cvt_15('F:\\zzxs\\Experiments\\dl-data\\ICDAR\\ICDAR2015\\sample_IC15\\train',
    #        'F:\\zzxs\\Experiments\\dl-data\\ICDAR\\ICDAR2015\\sample_IC15\\train_res')

    # cvt_curve('F:\\zzxs\\Experiments\\dl-data\\TotalText\\sample2',
    #           'F:\\zzxs\\Experiments\\dl-data\\TotalText\\res2', 'total', 1, True)

    # cvt_curve('F:\zzxs\Experiments\dl-data\CTW\ctw1500\sample',
    #           'F:\zzxs\Experiments\dl-data\CTW\ctw1500\\res', 'ctw1500', True)
##########################################



    # cvt_15('F:\\zzxs\\Experiments\\dl-data\\ICDAR\\ICDAR2015\\train',
    #        'F:\\zzxs\\Experiments\\dl-data\\ICDAR\\ICDAR2015\\train_cp', 1, False)
    #
    cvt_curve('../data/total/train',
              '../data/total/train_cp', 'total', 1, True)

    cvt_curve('../data/ctw1500/train',
              '../data/ctw1500/train_cp', 'ctw1500', 1, True)
