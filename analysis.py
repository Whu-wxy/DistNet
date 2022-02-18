import matplotlib.pyplot as plt
import numpy as np
import os

def readTxt(path):
	res = []
	if not os.path.exists(path):
		print('not exist.')
	else:
		print('exist.')

	with open(path, 'r', encoding='utf-8') as f:
		datas = f.readlines()
		for data in datas:
			res.append(list(map(float, data.split(','))))

	return res

def findFile(data1, data2, src_dir, dst_dir):
	for d1, d2 in zip(data1, data2):
		deltaP = d1[1] - d2[1]
		deltaR = d1[2] - d2[2]
		if deltaP * deltaR < 0:
			print(int(d1[0]), deltaP, deltaR)


data1 = readTxt('./ctw_0.4.txt')
data2 = readTxt('./ctw_0.8.txt')

findFile(data1, data2, None, None)