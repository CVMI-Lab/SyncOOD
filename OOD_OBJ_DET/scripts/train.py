import os
import h5py
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split
from .shared.datasets import FeatureDataset, collate_features
from .shared.metaclassifier import build_metaclassifier 

from tqdm import tqdm as tqdm

def compute_mean(file):
	mean = np.zeros((file["0"][:].shape[-1],))
	tally = 0
	for img_dets in tqdm(file.values()):
		mean += img_dets[:].sum(0)
		tally += img_dets[:].shape[0]
	mean /= tally
	file.create_dataset("mean", data=mean)
	return mean

def main(args):
	
	dset = "VOC" if "VOC" in args.config_file else "BDD"
	data_file = os.path.join(args.dataroot, f'{dset}_features/' + f'{dset}-{args.variant}-id.hdf5')
	ood_file = os.path.join(args.dataroot, f'{dset}_features/' + f'{dset}-{args.variant}-ood.hdf5')

	if dset == "VOC":
		mlp_config = {
			'lr': 0.001,
			'epochs': 5,
			'batch_size': 32,
			'optimizer': 'SGD',
			'ood_num': 4000,
			'ood_batch': 5,
		}
	elif dset == "BDD":
		mlp_config = {
			'lr': 0.00005,
			'epochs': 5,
			'batch_size': 32,
			'optimizer': 'SGD',
			'ood_num': 15000,
			'ood_batch': 64,
		}

	if args.random_seed is not None: torch.manual_seed(args.random_seed)
	generator = torch.Generator()

	## Load dataset
	h5file = h5py.File(data_file, 'r+')

	## Compute the dataset mean if it doesn't already exist
	if 'mean' in h5file.keys():
		means = h5file['mean'][:]
	else:
		print('Computing dataset mean...')
		means = compute_mean(h5file)

	means = torch.from_numpy(means).float().cuda()

	h5file.close()

	mlp_fname = os.path.join(args.dataroot, f'{dset}_features/' + f'{dset}-{args.variant}-mlp.pth')

	id_dataset = h5py.File(data_file, 'r+')
	ood_dataset = h5py.File(ood_file, 'r+')

	dataset = FeatureDataset(
		id_dataset=id_dataset,
		ood_dataset=ood_dataset,
		ood_num=mlp_config['ood_num'],
		ood_batch=mlp_config['ood_batch']
	)

	train_dataset, val_dataset = random_split(
		dataset, 
		[int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
		generator
	)

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=mlp_config['batch_size'],
		collate_fn=collate_features,
		shuffle=True,
		num_workers=8
	)

	val_dataloader = DataLoader(
		val_dataset,
		batch_size=mlp_config['batch_size'],
		collate_fn=collate_features,
		shuffle=False,
		num_workers=8
	)
	
	MLP, loss_fn, optimizer = build_metaclassifier(means.shape[0], mlp_config)
	MLP.train()
	MLP.cuda()


	train_MLP(
		train_dataloader,
		val_dataloader,
		MLP,
		loss_fn,
		optimizer,
		mlp_config,
		mlp_fname,
		means
	)

	id_dataset.close()
	ood_dataset.close()
	####################################
	## End Train Code
	####################################

def train_epoch(dataset, means, loss_fn, optimizer, MLP):
	MLP.train()
	loss_list = []
	total_pos_instance = 0
	for x, y in tqdm(dataset):
		x, y = x.cuda(), y.cuda()
		x -= means
		optimizer.zero_grad()
		y_hat = MLP(x).squeeze()
		loss = loss_fn(y_hat, y)
		loss_list.append(loss.item())

		total_pos_instance += torch.sum(y==1).item()

		loss.backward()
		optimizer.step()
	
	print('total_pos_instance', total_pos_instance)
		
	return torch.Tensor(loss_list).mean()

@torch.no_grad()
def val_epoch(dataset, means, loss_fn, MLP):
	MLP.eval()
	loss_list, acc, prec, rec = [], [], [], []

	for x, y in dataset:
		x, y = x.cuda(), y.cuda()
		x -= means
		y_hat = MLP(x).squeeze()
		
		loss = loss_fn(y_hat, y)
		loss_list.append(loss.item())

		preds = y_hat > 0.5
		true_pos = torch.logical_and(preds, y).sum()
		acc.append((y == preds).float().mean())
		prec.append(true_pos / preds.sum())
		rec.append(true_pos / y.sum())
	
	avg_loss = torch.Tensor(loss_list).mean()
	avg_acc = torch.Tensor(acc).mean()
	avg_prec = torch.Tensor(prec).mean()
	avg_rec = torch.Tensor(rec).mean()

	return avg_loss, avg_acc, avg_prec, avg_rec

def train_MLP(
		train_dataloader,
		val_dataloader,
		MLP,
		loss_fn,
		optimizer,
		config,
		mlp_fname,
		means,
	):
	best_loss = float('inf')
	for _ in tqdm(range(config['epochs'])):
		train_loss = train_epoch(train_dataloader, means, loss_fn, optimizer, MLP)
		val_loss, val_acc, prec, recall = val_epoch(val_dataloader, means, loss_fn, MLP)

		if train_loss < best_loss:
			best_loss = val_loss
			torch.save(MLP.state_dict(), mlp_fname)
		print(f'train_loss:{train_loss}')
		print(f'val_loss:{val_loss}')
		print(f'best_loss:{best_loss}')
		print(f'val_acc:{val_acc}')
		print(f'val_prec:{prec}')
		print(f'val_recall:{recall}')

	return MLP

def interface(args):
	main(args)


