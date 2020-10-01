import os

def	test_path(path):
	if not os.path.exists(path):
		raise ValueError("path is not exist!")

	files = os.listdir(path)
	prf_list = []
	for f_name in files:
		if len(prf_list) == 0:
			for line in open(os.path.join(path, f_name), 'r'):
				prf = line.split(',')
				prf_list.append([float(prf[0]), float(prf[1]), float(prf[2])])

				# p = prf[0]
				# r = prf[1]
				# f = prf[2]
				# F = (2 * p * r) / (p + r)
		else:
			for i, line in enumerate(open(os.path.join(path, f_name), 'r')):
				prf = line.split(',')
				cur_F = prf_list[i][2]
				if float(prf[2]) > cur_F:
					prf_list[i] = [float(prf[0]), float(prf[1]), float(prf[2])]

	p = r = f = 0
	for prf in prf_list:
		p += prf[0]
		r += prf[1]
		f += prf[2]
	p = p / len(prf_list)
	r = r / len(prf_list)
	f = f / len(prf_list)
	return p, r, f






if __name__=='__main__':

	path = './labels/total'
	p, r, f = test_path(path)
	print("precision: ", p)
	print("recall: ", r)
	print("f-measure: ", f)

	F = (2*p*r)/(p+r)
	print('F:, ', F)
