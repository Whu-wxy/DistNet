import os

basedir = 'F:\zzxs\Experiments\dl-data\TotalText\\test\gt'
files = os.listdir(basedir)
for file in files:
    with open(os.path.join(basedir, file), 'r') as f:
        for line in f.readlines():
            datas = line.split(',')
            if len(datas) % 2 != 1:
                print('not even:', file, ', end: ', datas[-1])
            points = []
            for i in datas[:-1]:
                if not i.isdigit():
                    # print('not digit:', i)
                    break
                else:
                    points.append(float(i))
            if len(points) % 2 != 0:
                print('not even:', file, ', end: ', datas[-1])


# basedir = 'F:\zzxs\Experiments\dl-data\CTW\ctw1500\\test\gt'
# files = os.listdir(basedir)
# for file in files:
#     with open(os.path.join(basedir, file), 'r') as f:
#         for line in f.readlines():
#             datas = line.split(',')
#             datas[-1] = datas[-1].strip()
#             if datas[-1] != 'CARE':
#                 print(file, ',', datas[-1])
