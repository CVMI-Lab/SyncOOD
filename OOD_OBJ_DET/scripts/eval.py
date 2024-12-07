"""
Probabilistic Detectron Inference Script
"""
import os

import core
import json
import sys
import torch
from tqdm import tqdm


from .shared import metric_utils as metrics, tracker as track, metaclassifier as meta, datasets as data
from functools import partial

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_ood_probabilistic_metrics
from inference.inference_utils import get_inference_output_dir, instances_to_json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
	dset = "VOC" if "VOC" in args.config_file else "BDD"

	data_file = os.path.join(args.dataroot, f'{dset}_features/' + f'{dset}-{args.variant}-id.hdf5')

	## Error checking
	if not "val" in args.test_dataset:
		raise ValueError('ERROR: Evaluating on non-testing set!')


	if "RCNN" in args.variant:
		from . import RCNN as model_utils
	else:
		from . import DETR as model_utils

	

	cfg = setup_config(args,
					   random_seed=args.random_seed,
					   is_testing=True)
	
	# Make sure only 1 data point is processed at a time. This simulates
	# deployment.
	cfg.defrost()

	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.MODEL.DEVICE = device.type

	# Set up number of cpu threads#
	torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

	####################################
	## Start Experiment Code
	####################################
	model, _, postprocessor = model_utils.build_model(
		cfg=cfg,
		args=args)
	
	tracker = track.featureTracker(model, args.variant)

	## Build metaclassifier for eval
	
	meta_classifier, means = meta.build_and_load_metaclassifier(args.mlp_path, data_file)
	meta_classifier.eval().cuda()

	## Define OOD scoring function
	ood_scoring = partial(detector_forward, MLP=meta_classifier.cuda().eval(), means=means) 
	
	# ## Load ID/OOD datasets
	cfgs, datasets, mappings, names = data.setup_test_datasets(args, cfg, model_utils)

	final_results = []
	for cfg, dataloader, mapping_dict, name in tqdm(zip(cfgs, datasets, mappings, names)):
		args.test_dataset = name

		print(f'Collecting scores for {name}...')

		# ####################################
		## Run inference
		######################################
		if args.variant == "DETR" and dset == "VOC":
			mapping_dict = model_utils.modify_voc_dict(mapping_dict)

		res = eval_dataset(
			predictor=model,
			dataloader=dataloader,
			tracker=tracker,
			mapping_dict=mapping_dict,
			postprocessors=postprocessor,
			model_utils=model_utils,
			ood_scorer=ood_scoring
		)

		####################################
		## Post processing
		####################################
		## Because of the way the COCO API functions, we cannot avoid using files for the average precision
		## This modifer helps us distinguish between separate runs of the same dataset
		
		output_dir = get_inference_output_dir(
			cfg['OUTPUT_DIR'],
			args.test_dataset,
			args.inference_config,
			args.image_corruption_level
		)

		## Error checking: output_dir directory may not exist on first run.
		if not os.path.exists(output_dir): os.makedirs(output_dir)

		with open(os.path.join(output_dir, f'coco_instances_results_save_{args.variant.upper()}.json'), 'w') as fp:
			json.dump(res, fp, indent=4, separators=(',', ': '))
		
		if len(final_results) < 1:
			if "RCNN" in args.variant:
				optimal_threshold = compute_average_precision.main_fileless(
					args,
					cfg,
					modifier=f"save_{args.variant.upper()}"
				)
				optimal_threshold = round(optimal_threshold, 4)
			else:
				optimal_threshold = 0.0

		processed_results = compute_ood_probabilistic_metrics.main_fileless(
			args,
			cfg,
			modifier=f"save_{args.variant.upper()}",
			min_allowed_score=optimal_threshold
		)
		
		final_results.append(processed_results)

		os.remove(os.path.join(output_dir, f'coco_instances_results_save_{args.variant.upper()}.json'))
		
	#####################################
	### Compute OOD performance metrics
	#####################################
	compute_metrics(final_results)


@torch.no_grad()
def detector_forward(tracker, boxes, outputs, MLP, means):
	## Perform ROI feature extraction
	mlp_input = tracker.roi_features([boxes], outputs.image_size[0])

	## Mean center the data
	mlp_input -= means

	## Remove the features from memory
	tracker.flush_features()

	## Perform inference pass with the metaclassifier MLP
	ood_scores = MLP(mlp_input).squeeze(-1)
	
	return -ood_scores


@torch.no_grad()
def eval_dataset(dataloader, predictor , tracker, mapping_dict, postprocessors, model_utils, ood_scorer):
	## Collect final outputs as determine by VOS benchmark
	final_output_list = []

	## iterate over the dataset
	for input_im in tqdm(dataloader):
		## Perform a forward pass with the DETR base model
		outputs, boxes, skip = model_utils.forward(predictor, input_im, postprocessors)
		
		## If there are no predicted boxes in the image, skip detection step.
		if not skip:
			## Retrieve OODness scores forall predicted boxes within the image.
			## Override the outputs.logistic_score value to carry the scores through to final evaluation. 
			ood_scores = ood_scorer(tracker, boxes, outputs)
			outputs.logistic_score = ood_scores

		#if len(final_output_list) > 1000: break
		## Add the detections as per the VOS benchmark.
		final_output_list.extend(
			instances_to_json(
				outputs,
				input_im[0]['image_id'],
				mapping_dict
			)
		)

	return final_output_list

def compute_metrics(results):
	id_score = torch.stack(results[0]['logistic_score']).cpu().numpy()
	coco_scores = torch.stack(results[1]['logistic_score']).cpu().numpy()
	open_scores = torch.stack(results[2]['logistic_score']).cpu().numpy()

	for ood_score, name in zip([coco_scores, open_scores], ['MS-COCO', 'OpenImages']):
		print(f'Metrics for {name}: ')
		measures = metrics.get_measures(-id_score, -ood_score, plot=False)
		metrics.print_measures(measures[0], measures[1], measures[2])



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


if __name__ == "__main__":
	# Create arg parser
	arg_parser = setup_arg_parser()

	args = arg_parser.parse_args()

	# Support single gpu inference only.
	args.num_gpus = 0
	# args.num_machines = 8

	print("Command Line Args:", args)

	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)

