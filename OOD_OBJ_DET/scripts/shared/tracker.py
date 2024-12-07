import torch.nn as nn
import torch
from torchvision.ops import roi_align
from functools import partial


class featureTracker():
	def __init__(self, model, variant='DETR'):
		self.variant = variant
		if "RCNN" in self.variant:
			model = model.model
		self.hook_model(model=model)


	@torch.no_grad()
	def __hook(self, model_self, inputs, outputs, idx):
		self.features[idx] = outputs

	@torch.no_grad()
	def hook_model(self, model):
		if self.variant == 'DETR':
			hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Sequential) and 'downsample' in n]
		elif self.variant == 'RCNN-RGX4':
			hook_queue = [
				model.backbone.bottom_up.s1.b1.bn,
				model.backbone.bottom_up.s2.b1.bn,
				model.backbone.bottom_up.s3.b1.bn,
				model.backbone.bottom_up.s4.b1.bn
			]
		elif self.variant == 'RCNN-RN50':
			hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and 'shortcut' in n]
		else:
			raise ValueError(f'Error: Target layers for model variant "{self.variant}" are not defined.')
		
		#print(f'Identified {len(hook_queue)} layers to target.')

		self.features = [0] * len(hook_queue)

		self.out_size = []
		for idx, module in enumerate(hook_queue):
			hook_fn = partial(self.__hook, idx=idx)
			module.register_forward_hook(hook_fn)
		
	@torch.no_grad()
	def roi_features(self, rois, input_h):
		det_feats = []
		for feat in self.features:
			_, _, h, _ = feat.size()
			scale = h/input_h
			feat = roi_align(feat, rois, (1, 1), scale).mean(dim=(2, 3))
			det_feats.append(feat)
		return torch.cat(det_feats, dim=1)

	def flush_features(self):
		self.features = [0] * len(self.features)
	
	def get_features(self):
		return self.features