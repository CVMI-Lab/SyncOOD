import torch
import h5py
from tqdm import tqdm
import os
import core
import sys
import json
from PIL import Image
import numpy as np

from tqdm import tqdm
import random
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

from detectron2.engine import launch
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.samplers.distributed_sampler import InferenceSampler

from core.setup import setup_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
	if "val" in args.test_dataset: 
		raise ValueError('Error: Feature extraction should only be performed on the training dataset to avoid accidental "training-on-test" errors.')
	cfg = setup_config(args,
					   random_seed=args.random_seed,
					   is_testing=True)
	cfg.defrost()
	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.MODEL.DEVICE = device.type

	# Set up number of cpu threads#
	torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)
	test_data_loader = build_detection_test_loader(cfg, dataset_name=args.test_dataset)
	cfg.INPUT.MIN_SIZE_TRAIN=800
	cfg.INPUT.RANDOM_FLIP='none'

	if "RCNN" in args.variant:
		from . import RCNN as model_utils
	else:
		from . import DETR as model_utils

	predictor, criterion, postprocessor = model_utils.build_model(
		cfg=cfg, 
		args=args
	)

	test_data_loader = build_detection_train_loader(
		cfg, 
		sampler=InferenceSampler(len(test_data_loader))
	)

	from .shared.tracker import featureTracker
	ConvTracker = featureTracker(predictor, args.variant)

	dset = "VOC" if "VOC" in args.config_file else "BDD"
	id_fname = f'{dset}-{args.variant}-id.hdf5'
	ood_fname = f'{dset}-{args.variant}-ood.hdf5'
	tmp_path = os.path.join(args.dataroot, f'{dset}_features')
	if not os.path.exists(tmp_path):
		os.makedirs(tmp_path)
	id_path = os.path.join(tmp_path, id_fname)
	ood_path = os.path.join(tmp_path, ood_fname)
	OOD_INFO_PATH = os.path.join(args.dataroot, f'SyncOOD_{dset}'+'/info.json')
	# with open(OOD_INFO_PATH, 'r') as file:
	# 	ood_info = json.load(file)

	capture_fn(
		dataloader=test_data_loader,
		model_utils=model_utils,
		predictor=predictor,
		tracker=ConvTracker,
		files=(id_path, ood_path),
		postprocessors=postprocessor,
		criterion=criterion,
		# syn_dataloader=ood_info
	)


def capture_fn(dataloader, model_utils, predictor, tracker, files, postprocessors, criterion, syn_dataloader=None):
	
	print()
	print('-----> Start to extract features of ID samples (' + str(len(dataloader.dataset)) + '):')
	id_file = h5py.File(files[0], 'w')
	for idx, input_im in enumerate(tqdm(dataloader)):
		input_im[0]['image'] = model_utils.channel_shift(input_im[0]['image'])
		kept_rois = extract_pass(
			input_im=input_im,
			predictor=predictor,
			postprocessors=postprocessors,
			model_utils=model_utils,
			tracker=tracker,
			dset_file=id_file,
			index=idx, 
			kept_rois=None
		)
	id_file.close()
	print('-----> Finish (extract features of ID samples) !')
	print()


	print('-----> Start to extract features of synthetic OOD samples:')
	ood_file = h5py.File(files[-1], 'w')
	if syn_dataloader is not None:
		for idx, input_im in enumerate(tqdm(syn_dataloader)):
			image = Image.open(input_im['file_name'])
			input_im['image'] = torch.tensor(np.array(image)).permute(2, 0, 1)  #.cuda()
			input_im['instances'] = torch.tensor(input_im['rois'])  #.cuda()
			kept_rois = torch.tensor(input_im['rois']).cuda()
			input_im = [input_im]
			_ = extract_pass_ood(
				input_im=input_im,
				predictor=predictor,
				postprocessors=postprocessors,
				model_utils=model_utils,
				tracker=tracker,
				dset_file=ood_file,
				index=idx, 
				kept_rois=kept_rois
			)
	ood_file.close()
	print('-----> Finish (extract features of synthetic OOD samples) !')
	print()


@torch.no_grad()
def extract_pass(input_im, predictor, postprocessors, model_utils, tracker, dset_file, index, kept_rois=None):
	h = input_im[0]['height']
	input_im[0]['image'] = model_utils.preprocess(input_im[0]['image']) if kept_rois is None else input_im[0]['image']
	_, boxes, _ = model_utils.forward(
		predictor=predictor,
		input_img=input_im,
		postprocessors=postprocessors,
	)
	kept_rois = boxes if kept_rois is None else kept_rois
	kept_rois = kept_rois.float()
	features = tracker.roi_features([kept_rois], h)
	features = features.detach().cpu().numpy()
	dset_file.create_dataset(f'{index}', data=features)
	tracker.flush_features()
	return kept_rois


@torch.no_grad()
def extract_pass_ood(input_im, predictor, postprocessors, model_utils, tracker, dset_file, index, kept_rois=None):
	h = input_im[0]['height']
	input_im[0]['image'] = model_utils.preprocess(input_im[0]['image']) if kept_rois is None else input_im[0]['image']
	_, boxes, _ = model_utils.forward(
		predictor=predictor,
		input_img=input_im,
		postprocessors=postprocessors,
	)
	kept_rois = kept_rois.float()
	features = tracker.roi_features([kept_rois], h)
	features = features.detach().cpu().numpy()
	dset_file.create_dataset(f'{index}', data=features)
	tracker.flush_features()
	del kept_rois
	del features
	return 1


def interface(args):
	print(args)
	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)