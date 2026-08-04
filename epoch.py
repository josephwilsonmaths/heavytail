# Script to train ResNet models and save eigenvalues / model metrics during training

from __future__ import annotations
import argparse
import json
import os
from pyexpat import model
import torch
import tqdm
import helpers
import math
import util.optimizers
import util.training
import util.classification_dataset
import util.networks
import os
import psutil

torch.set_default_dtype(torch.float32)


class RelabeledDataset(torch.utils.data.Dataset):
	def __init__(self, dataset: torch.utils.data.Dataset, labels: torch.Tensor) -> None:
		self.dataset = dataset
		self.labels = labels

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, index: int):
		X, _ = self.dataset[index]
		return X, self.labels[index]


def _extract_labels(dataset: torch.utils.data.Dataset) -> torch.Tensor:
	if isinstance(dataset, torch.utils.data.Subset):
		labels = _extract_labels(dataset.dataset)
		indices = torch.as_tensor(dataset.indices, dtype=torch.long)
		return labels[indices]

	for attr_name in ("targets", "labels"):
		if hasattr(dataset, attr_name):
			labels = getattr(dataset, attr_name)
			return torch.as_tensor(labels, dtype=torch.long)

	raise TypeError(f"Dataset of type {type(dataset).__name__} does not expose targets or labels.")


def with_uniform_random_labels(dataset: torch.utils.data.Dataset, n_classes: int) -> RelabeledDataset:
	"""Assigns each sample an i.i.d. uniform label from {0, ..., n_classes-1}."""
	random_labels = torch.randint(0, n_classes, (len(dataset),))
	return RelabeledDataset(dataset, random_labels)

def _evaluate(model, loader, loss_fn, device):
	loss, correct = 0.0, 0
	with torch.no_grad():
		for X, y in loader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	return loss / len(loader), correct / len(loader.dataset)


def print_ram_usage(tag=""):
    proc = psutil.Process(os.getpid())

    # Process RAM (this Python process)
    proc_rss_gb = proc.memory_info().rss / (1024 ** 3)

    # System RAM
    vm = psutil.virtual_memory()
    sys_used_gb = (vm.total - vm.available) / (1024 ** 3)
    sys_total_gb = vm.total / (1024 ** 3)

    print(
        f"{tag} Process RAM: {proc_rss_gb:.2f} GB | "
        f"System RAM: {sys_used_gb:.2f}/{sys_total_gb:.2f} GB ({vm.percent:.1f}%)"
    )

def main() -> None:
	print("Starting training runs...")

	parser = argparse.ArgumentParser(description="MiniAlexNet + CIFAR-10 loaders")
	parser.add_argument("--data_dir", type=str, default="data", help="Where CIFAR-10 is (or will be) downloaded")
	parser.add_argument("--model", type=str, default="minialexnet", help="Model architecture to use. Only 'minialexnet' is supported.")
	parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use. Only 'cifar10' is supported.")
	parser.add_argument("--subsample", action="store_true", help="Whether to subsample the dataset to 1000 examples for faster training")
	parser.add_argument("--num_repeats", type=int, default=3, help="Number of repeats to run for each batch size")
	parser.add_argument("--verbose", action="store_true", help="Whether to print training loss and accuracy at each epoch")
	parser.add_argument("--progress", action="store_true", help="Whether to show a tqdm progress bar during training")
	parser.add_argument("--batch_size", type=int, default=250, help="Batch size to use during training")
	parser.add_argument("--permuted_labels", action="store_true", help="Train on a random permutation of the training labels")
	parser.add_argument("--no_download", action="store_true", help="Require datasets to already exist locally instead of downloading them")
	args = parser.parse_args()
	n_epochs = 200
	learning_rate = 2 / args.batch_size
	weight_decay = 1e-4
	save_every = 5

	device = (
		"cuda"
		if torch.cuda.is_available()
		else "mps"
		if torch.backends.mps.is_available()
		else "cpu"
	)
	print(f"\n Using {device} device")
	if device == 'cuda':
		print(f"CUDA version: {torch.version.cuda}")

	batch_size = args.batch_size
	res_folder = 'results/htmp/epoch/'
	os.makedirs(res_folder, exist_ok=True)
	config = {
		"model": args.model,
		"dataset": args.dataset,
		"data_dir": args.data_dir,
		"subsample": args.subsample,
		"num_repeats": args.num_repeats,
		"batch_size": batch_size,
		"permuted_labels": args.permuted_labels,
		"no_download": args.no_download,
		"epochs": n_epochs,
		"save_every": save_every,
		"optimizer": "sgd",
		"learning_rate": learning_rate,
		"weight_decay": weight_decay,
		"device": device,
	}
	config_filename = f'{args.model}_{args.dataset}'
	if args.subsample:
		config_filename += '_subsampled'
	if args.permuted_labels:
		config_filename += '_permuted_labels'
	with open(os.path.join(res_folder, f'{config_filename}_config.json'), 'w', encoding='utf-8') as config_file:
		json.dump(config, config_file, indent=2)
	

	for repeat in range(args.num_repeats):
		print(f'Repeat {repeat+1}/{args.num_repeats} for batch size {batch_size}')

		# Get dataset
		dataset = util.classification_dataset.load_dataset(
			name=args.dataset,
			subsample=args.subsample,
			augment=False,
			shuffle=False,
			data_dir=args.data_dir,
			download=not args.no_download,
			include_ood=False,
		)
		# ---

		print('Creating dataloader')
		train_dataset = dataset.training_data
		if args.permuted_labels:
			train_dataset = with_uniform_random_labels(train_dataset, dataset.n_output)
			print('Training with permuted labels')

		# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=dataset.shuffle)
		# test_loader = dataset.testloader(batch_size=batch_size)

		if hasattr(os, "sched_getaffinity"):
			available_cpus = len(os.sched_getaffinity(0))
		else:
			available_cpus = os.cpu_count() or 1

		num_workers = min(8, available_cpus)

		loader_kwargs = {
			"batch_size": batch_size,
			"num_workers": num_workers,
			"pin_memory": device == "cuda",
			"persistent_workers": num_workers > 0,
		}

		if num_workers > 0:
			loader_kwargs["prefetch_factor"] = 4

		train_loader = torch.utils.data.DataLoader(
			train_dataset,
			shuffle=dataset.shuffle,
			**loader_kwargs,
		)

		if args.permuted_labels:
			# independent draw — p(y|x) = U({0,...,C-1}), no shared mapping with training
			_permuted_test_data = with_uniform_random_labels(dataset.test_data, dataset.n_output)
			test_loader = torch.utils.data.DataLoader(dataset.test_data, shuffle=False, **loader_kwargs)
			permuted_test_loader = torch.utils.data.DataLoader(_permuted_test_data, shuffle=False, **loader_kwargs)
		else:
			_n_test = len(dataset.test_data)
			_n_val = int(0.2 * _n_test)
			_test_split, _val_split = torch.utils.data.random_split(
				dataset.test_data,
				[_n_test - _n_val, _n_val],
				generator=torch.Generator().manual_seed(0),
			)
			test_loader = torch.utils.data.DataLoader(_test_split, shuffle=False, **loader_kwargs)
			val_loader = torch.utils.data.DataLoader(_val_split, shuffle=False, **loader_kwargs)
		loss_fn = torch.nn.CrossEntropyLoss()

		model = util.networks.get_model(args.model, num_classes=dataset.n_output, n_channels=dataset.n_channels).to(device)

		optimizer, scheduler = util.optimizers.get_optim_sched(model, 'sgd', None, learning_rate, weight_decay, n_epochs)

		for epoch in tqdm.trange(n_epochs):
			model.train()
			size = len(train_loader.dataset)
			num_batches = len(train_loader)    
			train_loss, train_correct = 0, 0

			if args.verbose and args.progress:
				pbar_inner = tqdm.tqdm(enumerate(train_loader))
			else:
				pbar_inner = enumerate(train_loader)
			for batch, (X, y) in pbar_inner:
				X,y = X.to(device), y.to(device)
				# Compute prediction and loss
				pred = model(X)
				loss = loss_fn(pred, y)

				# Backpropagation
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				# Evaluate metrics
				train_loss += loss
				train_correct += (pred.argmax(1) == y).type(torch.float).sum()

			train_loss = (train_loss / len(train_loader)).item()
			train_acc = (train_correct / len(train_loader.dataset)).item()

			if args.verbose:
				print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

			if math.isnan(train_loss):
				print("Training loss is NaN, skipping this run.")
				continue

			if epoch % save_every == 0:
				model.eval()
				test_loss, test_acc = _evaluate(model, test_loader, loss_fn, device)

				if args.permuted_labels:
					permuted_test_loss, permuted_test_acc = _evaluate(model, permuted_test_loader, loss_fn, device)
					print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
						  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
						  f"Perm Test Loss: {permuted_test_loss:.4f}, Perm Test Acc: {permuted_test_acc:.4f}")
				else:
					val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device)
					print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
						  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
						  f"Val Acc: {val_acc:.4f}")

				weight_spectra = helpers.weight_spectra_wtw(model)

				res_dict = {
					"batch_size": batch_size,
					"epoch": epoch,
					"repeat": repeat,
					"train_loss": train_loss,
					"train_acc": train_acc,
					"test_loss": test_loss,
					"test_acc": test_acc,
					"weight_spectra_wtw": weight_spectra,
				}
				if args.permuted_labels:
					res_dict["permuted_test_loss"] = permuted_test_loss
					res_dict["permuted_test_acc"] = permuted_test_acc
				else:
					res_dict["val_loss"] = val_loss
					res_dict["val_acc"] = val_acc

				filename = f'{args.model}_{args.dataset}'
				if args.subsample:
					filename += '_subsampled'
				if args.permuted_labels:
					filename += '_permuted_labels'
				filename += f'_b{batch_size}_{epoch}_{repeat}.pt'

				torch.save(res_dict, res_folder + filename)
				print(f"Saved results to {res_folder + filename}\n")

if __name__ == "__main__":
	main()
