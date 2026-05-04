# Script to train ResNet models and save eigenvalues / model metrics during training

from __future__ import annotations
import argparse
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

torch.set_default_dtype(torch.float64)

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
	args = parser.parse_args()

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
	

	for repeat in range(args.num_repeats):
		print(f'Repeat {repeat+1}/{args.num_repeats} for batch size {batch_size}')

		# Get dataset
		dataset = util.classification_dataset.load_dataset(name=args.dataset, subsample=args.subsample, augment=False, shuffle=False)
		# ---

		print('Creating dataloader')
		train_loader = dataset.trainloader(batch_size=batch_size)
		test_loader = dataset.testloader(batch_size=batch_size)
		loss_fn = torch.nn.CrossEntropyLoss()

		model = util.networks.get_model(args.model, num_classes=dataset.n_output, n_channels=dataset.n_channels).to(device)

		optimizer, scheduler = util.optimizers.get_optim_sched(model, 'sgd', None, 2 / batch_size, 1e-4, 200)

		for epoch in tqdm.trange(200):
			model.train()
			size = len(train_loader.dataset)
			num_batches = len(train_loader)    
			train_loss, train_acc = 0, 0
			for batch, (X, y) in enumerate(train_loader):
				X,y = X.to(device), y.to(device)
				# Compute prediction and loss
				pred = model(X)
				loss = loss_fn(pred, y)

				# Backpropagation
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				# Evaluate metrics
				train_loss += loss.item()
				train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

			train_loss /= num_batches
			train_acc /= size

			if args.verbose:
				print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

			if math.isnan(train_loss):
				print("Training loss is NaN, skipping this run.")
				continue

			if epoch % 5 == 0:
				model.eval()
				test_loss, test_acc = 0, 0
				with torch.no_grad():
					for X, y in test_loader:
						X,y = X.to(device), y.to(device)
						pred = model(X)
						test_loss += loss_fn(pred, y).item()
						test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

				test_loss /= len(test_loader)
				test_acc /= len(test_loader.dataset)
		
				print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

				# Weights.
				K = helpers.weight_gram_sl(model)
				weight_eigvals = torch.linalg.eigvalsh(K)
				weight_F, weight_trace, weight_logdet = helpers.free_energy_eigs(weight_eigvals, K.shape[0], lam=1, jitter=0)

				print(f'Weights: F: {weight_F:.4f}, G: {weight_trace:.4f}, LD: {weight_logdet:.4f}')

				res_dict = {
					"batch_size": batch_size,
					"train_loss": train_loss,
					"train_acc": train_acc,
					"test_loss": test_loss,
					"test_acc": test_acc,
					"weight_free_energy": weight_F,
					"weight_eigvals": weight_eigvals.cpu().numpy(),
				}

				del(K); del(weight_eigvals)

				filename = f'{args.model}_{args.dataset}'
				if args.subsample:
					filename += '_subsampled'
				filename += f'_b{batch_size}_{epoch}_{repeat}.pt'

				res_folder = 'results/htmp/epoch/'

				os.makedirs(res_folder, exist_ok=True)

				torch.save(res_dict, res_folder + filename)
				print(f"Saved results to {res_folder + filename}\n")

if __name__ == "__main__":
	main()
