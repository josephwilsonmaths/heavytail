import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import os

def _dataset_root(data_dir, dataset_name):
    normalized_dir = os.path.normpath(data_dir)
    if os.path.basename(normalized_dir).lower() == dataset_name.lower():
        return normalized_dir
    return os.path.join(normalized_dir, dataset_name)


def _validate_dataset_root(root, download, dataset_name):
    if not download and not os.path.isdir(root):
        raise FileNotFoundError(
            f"{dataset_name} dataset directory was not found at '{root}'. "
            "Pass a valid --data_dir or omit --no_download to allow fetching it."
        )


def _load_with_optional_download(dataset_ctor, dataset_name, root, download, **kwargs):
    try:
        return dataset_ctor(root=root, download=download, **kwargs)
    except RuntimeError as exc:
        if not download:
            raise FileNotFoundError(
                f"{dataset_name} data was not ready under '{root}'. "
                "Pass a directory containing the extracted torchvision dataset files, "
                "or omit --no_download to allow fetching them."
            ) from exc
        raise


def get_dataset(name, subsample, augment=False, data_dir="data", download=True, include_ood=True):
    if name=='mnist':
        training_data = datasets.MNIST(
            root=_dataset_root(data_dir, "MNIST"),
            train=True,
            download=download,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root=_dataset_root(data_dir, "MNIST"),
            train=False,
            download=download,
            transform=ToTensor()
        )

        ood_test_data = None
        if include_ood:
            ood_test_data = datasets.FashionMNIST(
                root=_dataset_root(data_dir, "FashionMNIST"),
                train=False,
                download=download,
                transform=ToTensor()
            )

        _, val_data = torch.utils.data.random_split(training_data,[50000,10000])
        n_output = 10
        n_channels = 1
        input_size = 28

    elif name=='fmnist':
        training_data = datasets.FashionMNIST(
            root=_dataset_root(data_dir, "FashionMNIST"),
            train=True,
            download=download,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=_dataset_root(data_dir, "FashionMNIST"),
            train=False,
            download=download,
            transform=ToTensor()
        )

        ood_test_data = None
        if include_ood:
            ood_test_data = datasets.MNIST(
                root=_dataset_root(data_dir, "MNIST"),
                train=False,
                download=download,
                transform=ToTensor()
            ) 

        _, val_data = torch.utils.data.random_split(training_data,[50000,10000])
        n_output = 10
        n_channels = 1
        input_size = 28

    elif name=='cifar10':
        cifar10_root = _dataset_root(data_dir, "CIFAR10")
        _validate_dataset_root(cifar10_root, download, "CIFAR10")
        training_data = _load_with_optional_download(
            datasets.CIFAR10,
            "CIFAR10",
            cifar10_root,
            download,
            train=True,
            transform=transforms.ToTensor()
        )

        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            data = torch.as_tensor(training_data.data).permute(0, 3, 1, 2).to(torch.float32) / 255.0
            mean = data.mean(dim=(0, 2, 3))
            std = data.std(dim=(0, 2, 3))
            d = data[0].numel()  # 3*32*32 = 3072

            # Define data transforms
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist()),
                transforms.Lambda(lambda x: x * (np.sqrt(d) / torch.norm(x.view(-1)))),
            ])
            
        if augment:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_test = transform_train

        training_data = _load_with_optional_download(
            datasets.CIFAR10,
            "CIFAR10",
            cifar10_root,
            download,
            train=True,
            transform=transform_train
        )

        test_data = _load_with_optional_download(
            datasets.CIFAR10,
            "CIFAR10",
            cifar10_root,
            download,
            train=False,
            transform=transform_test
        )

        ood_test_data = None
        if include_ood:
            ood_test_data = datasets.CIFAR100(
                root=_dataset_root(data_dir, "CIFAR100"),
                train=False,
                download=download,
                transform=transform_test
            ) 

        _, val_data = torch.utils.data.random_split(training_data,[40000,10000])
        n_output = 10
        n_channels = 3
        input_size = 32

    elif name=='svhn':
        svhn_root = _dataset_root(data_dir, "SVHN")
        _validate_dataset_root(svhn_root, download, "SVHN")
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])

        training_data = datasets.SVHN(
            root=svhn_root,
            split='train',
            download=download,
            transform=transform_train
        )

        test_data = datasets.SVHN(
            root=svhn_root,
            split='test',
            download=download,
            transform=transform_test
        )

        transform_test_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ood_test_data = None
        if include_ood:
            ood_test_data = datasets.CIFAR10(
                root=_dataset_root(data_dir, "CIFAR10"),
                train=False,
                download=download,
                transform=transform_test_cifar
            ) 

        _, val_data = torch.utils.data.random_split(training_data,[0.9,0.1])
        n_output = 10
        n_channels = 3
        input_size = 32

    elif name=='cifar100':
        cifar100_root = _dataset_root(data_dir, "CIFAR100")
        _validate_dataset_root(cifar100_root, download, "CIFAR100")
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        training_data = datasets.CIFAR100(
            root=cifar100_root,
            train=True,
            download=download,
            transform=transform_train
        )

        test_data = datasets.CIFAR100(
            root=cifar100_root,
            train=False,
            download=download,
            transform=transform_test
        )

        ood_test_data = None
        if include_ood:
            ood_test_data = datasets.CIFAR10(
                root=_dataset_root(data_dir, "CIFAR10"),
                train=False,
                download=download,
                transform=transform_test
            ) 

        val_data = test_data
        n_output = 100
        n_channels = 3
        input_size = 32

    elif name == 'imagenet':
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        
        training_data = datasets.ImageFolder(
            root='/scratch/licenseddata/imagenet/imagenet-1k/train',
            transform = preprocess
        )

        mean_test = [0.485, 0.456, 0.406]
        std_test = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose(
            [transforms.Resize(232), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean_test, std_test)])

        test_data = datasets.ImageFolder(
            root='/scratch/licenseddata/imagenet/imagenet-1k/val',
            transform = test_transform
        )

        mean_ood = [0.485, 0.456, 0.406]
        std_ood = [0.229, 0.224, 0.225]

        ood_transform = transforms.Compose(
            [transforms.Resize(232), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean_ood, std_ood)])

        print('loading datasets')

        ood_test_data = None
        if include_ood:
            ood_test_data = datasets.ImageFolder(os.path.join(data_dir, 'imagenet-o', 'imagenet-o'), transform=ood_transform)

        n_output = 1000
        n_channels = 3
        input_size = 224

        val_data = test_data

        print('loaded datasets')

    if subsample:
        n_train = 1000
        n_test = 1000
        training_data = torch.utils.data.Subset(training_data,range(n_train))
        test_data = torch.utils.data.Subset(test_data,range(n_test))
        val_data = torch.utils.data.Subset(val_data,range(n_test))
        if ood_test_data is not None:
            ood_test_data = torch.utils.data.Subset(ood_test_data,range(n_test))

    return training_data, test_data, val_data, ood_test_data, n_output, n_channels, input_size

class load_dataset(object):
    def __init__(self, name, subsample, augment=False, shuffle=False, data_dir="data", download=True, include_ood=True):
        training_data, test_data, val_data, ood_test_data, n_output, n_channels, input_size = get_dataset(
            name=name,
            subsample=subsample,
            augment=augment,
            data_dir=data_dir,
            download=download,
            include_ood=include_ood,
        )
        self.training_data = training_data
        self.test_data = test_data
        self.val_data = val_data
        self.ood_test_data = ood_test_data
        self.n_output = n_output
        self.n_channels = n_channels
        self.input_size = input_size
        self.shuffle = shuffle

    def trainloader(self, batch_size):
        return DataLoader(self.training_data, batch_size=batch_size, shuffle=self.shuffle)
    
    def testloader(self, batch_size):
        return DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
    
    def valloader(self, batch_size):
        return DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
    
    def oodtestloader(self, batch_size):
        if self.ood_test_data is None:
            raise ValueError("OOD dataset was not loaded for this dataset instance.")
        return DataLoader(self.ood_test_data, batch_size=batch_size, shuffle=False)

    

