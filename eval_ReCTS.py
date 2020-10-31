import torch
import shutil
import numpy as np
import config
import os
import cv2
from tqdm import tqdm
from models import FPN_ResNet
from torchvision import transforms
from predict_ic15 import Pytorch_model
from utils import draw_bbox
from dist import decode as dist_decode
import timeit
import Polygon

from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

torch.backends.cudnn.benchmark = True

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

def write_result_as_txt(save_path, bboxes, scores=None):
	lines = []

	covered_list = []
	uncovered_list = []

	res_boxes = []
	for (i, box) in enumerate(bboxes):
		if i in covered_list:
			continue

		p = Polygon.Polygon(box)
		for j, tempBox in enumerate(bboxes):
			if j == i:
				continue
			if j in covered_list:
				continue
			p2 = Polygon.Polygon(tempBox)
			if p.covers(p2):
				covered_list.append(j)

	for b_idx, bbox in enumerate(bboxes):
		line = ''
		if b_idx in covered_list:
			continue

		for box in bbox:
			res_boxes.append(bbox)
			line += "%d, %d, "%(max(int(box[0]), 0), max(int(box[1]),0) )

		score = 1
		if scores != None:
			if b_idx>len(scores)-1:
				score = 1
			else:
				score = scores[b_idx]
			line += str(score)
		else:
			line = line.rsplit(',', 1)[0]
		line += '\n'
		lines.append(line)
	with open(save_path, 'w') as f:
		for line in lines:
			f.write(line)

	return  np.array(res_boxes)


class Pytorch_model_ReCTS:
	def __init__(self, model_path, net, scale, gpu_id=None):
		'''
		初始化pytorch模型
		:param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
		:param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
		:param img_channel: 图像的通道数: 1,3
		:param gpu_id: 在哪一块gpu上运行
		'''
		self.scale = scale
		if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
			self.device = torch.device("cuda:{}".format(gpu_id))
		else:
			self.device = torch.device("cpu")

		#self.net = net.to(self.device)
		self.net = torch.load(model_path, map_location=self.device)['state_dict']
		#self.net = torch.jit.load(model_path, map_location=self.device)

		print('device:', self.device)

		if net is not None:
			# 如果网络计算图和参数是分开保存的，就执行参数加载
			net = net.to(self.device)
			net.scale = scale
			try:
				sk = {}
				for k in self.net:
					sk[k[7:]] = self.net[k]
				net.load_state_dict(sk)
			except:
				net.load_state_dict(self.net)
			self.net = net
			print('load models')
		self.net.eval()

	def predict(self, img_path: str, long_size: int = 2240, fast_test=True):
		'''
		对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
		:param img: 图像地址
		:param is_numpy:
		:return:
		'''
		assert os.path.exists(img_path), 'file is not exists'
		img = None
		try:
			if img_path.endswith('jpg'):
				in_file = open(img_path, 'rb')
				img = jpeg.decode(in_file.read())
				in_file.close()
				# im = jpeg.JPEG(im_fn).decode()
			else:
				img = cv2.imread(img_path)
		except:
			print('error: ', img_path)

		try:
			#img = np.asarray(img, dtype=np.uint8)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		except:
			print('error: ', img_path)
		h, w = img.shape[:2]

		if long_size != None:
			scale = long_size / max(h, w)
			scale = 1.5
			img = cv2.resize(img, None, fx=scale, fy=scale)

		# 将图片由(w,h)变为(1,img_channel,h,w)
		tensor = transforms.ToTensor()(img)
		tensor = tensor.unsqueeze_(0)

		tensor = tensor.to(self.device)
		with torch.no_grad():
			if torch.cuda.is_available():
				torch.cuda.synchronize()

			model_time = timeit.default_timer()
			preds = self.net(tensor)
			model_time = (timeit.default_timer() - model_time)

			decode_time = timeit.default_timer()
			res_preds, boxes_list, scores_list = dist_decode(preds[0], scale)
			decode_time = (timeit.default_timer() - decode_time)

			if not fast_test:
				decode_time = timeit.default_timer()
				for i in range(50):  # same as DBNet: https://github.com/MhLiao/DB/blob/master/eval.py
					preds_temp, boxes_list, scores_list = dist_decode(preds[0], scale)
				decode_time = (timeit.default_timer() - decode_time) / 50.0

			t = model_time + decode_time

			scale = (res_preds.shape[1] / w, res_preds.shape[0] / h)
			if len(boxes_list):
				boxes_list = boxes_list / scale
		return res_preds, boxes_list, t, scores_list, model_time, decode_time  #, logit


def main(net, model_path, long_size, scale, path, save_path, gpu_id, fast_test):
	if os.path.exists(save_path):
		shutil.rmtree(save_path, ignore_errors=True)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	save_img_folder = os.path.join(save_path, 'img')
	if not os.path.exists(save_img_folder):
		os.makedirs(save_img_folder)
	save_txt_folder = os.path.join(save_path, 'result')
	if not os.path.exists(save_txt_folder):
		os.makedirs(save_txt_folder)
	img_paths = [os.path.join(path, x) for x in os.listdir(path)]

	model = Pytorch_model_ReCTS(model_path, net=net, scale=scale, gpu_id=gpu_id)
	total_frame = 0.0
	total_time = 0.0
	model_total_time = 0.0
	decode_total_time = 0.0
	for img_path in tqdm(img_paths):
		img_name = os.path.basename(img_path).split('.')[0]
		#lab_name = 'res_img_' + img_name.split('_')[-1]
		lab_name = img_name
		save_name = os.path.join(save_txt_folder, lab_name + '.txt')

		if os.path.exists(save_name):
			continue

		pred, boxes_list, t, scores_list, model_time, decode_time = model.predict(img_path, long_size=long_size, fast_test=fast_test)
		total_frame += 1
		total_time += t
		model_total_time += model_time
		decode_total_time += decode_time

		res_boxes = write_result_as_txt(save_name, boxes_list)

		if isinstance(img_path, str):
			text_box = cv2.imread(img_path)
		text_box = draw_bbox(img_path, res_boxes, color=(0, 255, 0), thickness=4)
		cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), text_box)


	print('fps:{}'.format(total_frame / total_time))
	print('average model time:{}'.format(model_total_time / total_frame))
	print('average decode time:{}'.format(decode_total_time / total_frame))
	return save_txt_folder


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
	long_size = 2000     #2240
	scale = 1    #DistNet_IC17_130_loss1.029557.pth
	model_path = '../save/ReCTS/dist_ReCTS_mobile_ours2/DistNet_IC17_100_loss0.569300.pth'   #DistNet_IC17_97_loss1.057110.pth
	#DistNet_ReCTS_188_loss0.407465.pth
	#../save/ReCTS/dist_ReCTS/DistNet_ReCTS_155_loss0.288792.pth
	#../save/ReCTS/dist_ReCTS_mobile/DistNet_ReCTS_188_loss0.407465.pth
	#../save/ReCTS/dist_ReCTS_mobile_ours/DistNet_IC17_121_loss0.564716.pth
	#../save/ReCTS/dist_ReCTS_mobile_ours2/DistNet_IC17_100_loss0.569300.pth
	data_path = '../data/ReCTS/ReCTS_OUR/test/img'   #../data/ReCTS/test/img/ghz  ../data/ReCTS/ReCTS_OUR/test
	save_path = '../test_resultReCTS'
	gpu_id = 0
	print('model_path:{}'.format(model_path))

	fast_test=True

	from models.craft import CRAFT
	from models.mobilenetv3_fpn import mobilenetv3_fpn

	#net = CRAFT(num_out=2, pretrained=False, scale=scale)
	net = mobilenetv3_fpn(num_out=2)

	save_path = main(net, model_path, long_size, scale, data_path, save_path, gpu_id=gpu_id, fast_test=fast_test)

