from numba import njit, jit, set_num_threads, prange
from numba.typed import List

import numpy as np
import timeit

set_num_threads(8)

@jit(parallel=True, nopython=True, cache=True)
def test(x):
    n = x.shape[0]
    a = np.sin(x)
    b = np.cos(a * a)
    acc = 0
    for i in prange(n - 2):
        for j in prange(n - 1):
            acc += b[i] + b[j + 1]
    return acc

@jit(parallel=True, nopython=True, cache=True)
def test2(region_label_img, center_label_img, lab):
	aa = np.where(center_label_img == lab)

	return region_label_img[aa[0][0]][aa[1][0]]

	# res = 0
	# num = 0
	# m, n = center_label_img.shape[0], center_label_img.shape[1]
	# for i in prange(m):
	# 	for j in range(n):
	# 		if res != 0:
	# 			continue
	# 		if center_label_img[i][j] == lab:
	# 			res += region_label_img[i][j]
	# 			num += 1
	# return res//num

# test(np.arange(10))

a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
b = np.array([[2, 2, 7, 9], [2, 7, 0, 2], [3, 3, 3, 3]])
c = np.array([9])

t1 = timeit.default_timer()
res = test2(a, b, 7)
print('res: ', res)
print('time: ', timeit.default_timer()-t1)
test2.parallel_diagnostics(level=4)


