
import torch
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

class Recognizer(object):
	def __init__(self, ckpt_file, exp, fp16=False):
		self.preproc = ValTransform()

		self.exp = exp
		model = exp.get_model()
		model.cuda()
		model.eval()
		
		ckpt = torch.load(ckpt_file, map_location="cpu")
		model.load_state_dict(ckpt["model"])

		self.model = model

		self.fp16 = fp16
		self.decoder = None

		self.confthre = exp.test_conf

		pass

	def inference(self, img):
		height, width = img.shape[:2]
		ratio = min(self.exp.test_size[0] / img.shape[0], self.exp.test_size[1] / img.shape[1])
		img, _ = self.preproc(img, None, self.exp.test_size)
		img = torch.from_numpy(img).unsqueeze(0)
		img = img.float().cuda()
		if self.fp16:
			img = img.half()  # to FP16
		
		with torch.no_grad():
			outputs = self.model(img)
			if self.decoder is not None:
				outputs = self.decoder(outputs, dtype=outputs.type())

			outputs = postprocess(
					outputs, self.exp.num_classes, self.confthre,
					self.exp.nmsthre, class_agnostic=True
				)
		
		output = outputs[0].cpu()
		bboxes = output[:, 0:4]
		bboxes /= ratio
		scores = output[:, 4] * output[:, 5]
		cls = output[:, 6]

		return (cls, scores, bboxes)