import torch
import numpy as np

from detectron2.utils.events import EventStorage
from inference.inference_utils import build_predictor

from torch.autograd import Variable


###############################
#####					  #####
##### 	 PREPROCESSORS	  #####
#####					  #####
###############################

### RCNN does not need the same degree of preprocessing that DETR does.
def preprocess(a): 		return a
def channel_shift(a): 	return a
def modify_voc_dict(a):	return a

## Preprocessing mapping is already defined for RCNN
def get_mapper(): return None





###############################
#####					  #####
##### 		BUILDER 	  #####
#####					  #####
###############################

## Model builder for RCNN is already implemented in VOS.
def build_model(cfg, **kwargs):
	predictor = build_predictor(cfg)
	predictor.model.cuda()
	predictor.model.eval()
	return predictor, None, None





###############################
#####					  #####
##### 		INFERENCE	  #####
#####					  #####
###############################

@torch.no_grad()
def forward(predictor, input_img, *args, **kwargs):
	## Perform forward pass over the input image
	outputs = predictor(input_img)
	boxes = outputs.pred_boxes
	#print(len(boxes))
	#print(outputs)
	#exit()
	## Return the newly formatted outputs, the predicted regions of interest and a skip signifier
	return outputs, boxes.tensor, len(boxes) < 1 

	


