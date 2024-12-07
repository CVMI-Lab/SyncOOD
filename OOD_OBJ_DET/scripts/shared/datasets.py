from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from detectron2.data import build_detection_test_loader, MetadataCatalog
from core.setup import setup_config
from torch.utils.data import Dataset

import torch
import numpy as np
import random
import os

def setup_test_datasets(args, cfg, variant):
	coco_name = 'coco_ood_val{}'.format('_bdd' if 'BDD' in args.config_file else '')
	names = [args.test_dataset, coco_name, 'openimages_ood_val']
	dirs = [args.dataset_dir, os.path.join(args.dataroot, 'COCO/'), os.path.join(args.dataroot, 'OpenImages/')]
	cfgs, datasets, map_dicts = [], [], []

	for idx, (name, direct) in enumerate(zip(names, dirs)):
		args.test_dataset = name
		args.dataset_dir = direct
		if idx:
			cfg = setup_config(args,
				random_seed=args.random_seed,
				is_testing=True
			)
		data_loader = build_detection_test_loader(
			cfg, 
			mapper=variant.get_mapper(),
			dataset_name=args.test_dataset
		)

		train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
			cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
		
		test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
			args.test_dataset).thing_dataset_id_to_contiguous_id

		# If both dicts are equal or if we are performing out of distribution
		# detection, just flip the test dict.
		cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
			cfg,
			args,
			train_thing_dataset_id_to_contiguous_id,
			test_thing_dataset_id_to_contiguous_id
		)
		cfgs.append(cfg)
		datasets.append(data_loader)
		map_dicts.append(cat_mapping_dict)

	return cfgs, datasets, map_dicts, names

	
class FeatureDataset(Dataset):
	def __init__(self, id_dataset, ood_dataset, ood_num, ood_batch):
		self.id_dataset = id_dataset
		self.ood_dataset = ood_dataset
		self.ood_num = ood_num
		self.ood_batch = ood_batch
		self.ood_idx_pool = list(range(len(self.ood_dataset)))
		random.shuffle(self.ood_idx_pool)
		self.ood_idx_pool = self.ood_idx_pool[:self.ood_num]

	def __len__(self):
		return len(self.id_dataset.keys())-1

	def __getitem__(self, idx):
		id_sample = self.id_dataset[f'{idx}'][:]
		new_idx = random.randint(0, self.ood_num - 1)
		ood_sample = self.ood_dataset[f'{self.ood_idx_pool[new_idx]}'][:]

		for i_add in range(self.ood_batch - 1):
			new_idx = random.randint(0, self.ood_num - 1)
			ood_sample1 = self.ood_dataset[f'{self.ood_idx_pool[new_idx]}'][:]
			ood_sample = np.concatenate((ood_sample, ood_sample1), axis=0)

		data = np.concatenate((id_sample, ood_sample), axis=0)
		
		labels = np.ones(len(id_sample)+len(ood_sample))
		labels[len(id_sample):] = 0

		return data, labels

def collate_features(data):
	x_list = np.concatenate([d[0] for d in data], axis=0)
	y_list = np.concatenate([d[1] for d in data], axis=0)
	return torch.from_numpy(x_list).float(), torch.from_numpy(y_list).float()

