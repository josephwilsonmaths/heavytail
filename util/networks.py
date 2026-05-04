import util.training
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models import *

def get_model(name, n_channels, num_classes=10, num_p=0):
	if name == 'lenet':
		network = LeNet(n_channels=n_channels, num_classes=num_classes)
	elif name == 'minialexnet':
		network = MiniAlexNet(n_channels=n_channels, num_classes=num_classes)
	elif name == 'resnet9':
		network = ResNet9(in_channels=n_channels, num_classes=num_classes, p=num_p)
	elif name == 'resnet18':
		network = ResNet18(in_channels=n_channels, num_classes=num_classes, p=num_p)
	elif name == 'resnet34':
		network = ResNet34(in_channels=n_channels, num_classes=num_classes, p=num_p)
	elif name == 'resnet50':
		network = ResNet50(in_channels=n_channels, num_classes=num_classes, p=num_p)

	return network

class LeNet(nn.Module):
	def __init__(self, n_channels=3, num_classes=10):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(n_channels, 6, 5, padding=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.pool = nn.AvgPool2d(2, stride=2)
		self.flat = nn.Flatten()
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, num_classes)

		self.reset_parameters()

	def reset_parameters(self) -> None:
		"""Initialize trainable weights i.i.d. N(0, 1/sqrt(fan_in))."""
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
				with torch.no_grad():
					module.weight.normal_(mean=0.0, std=1.0 / math.sqrt(fan_in))
					if module.bias is not None:
						module.bias.zero_()
			elif isinstance(module, nn.Linear):
				fan_in = module.in_features
				with torch.no_grad():
					module.weight.normal_(mean=0.0, std=1.0 / math.sqrt(fan_in))
					if module.bias is not None:
						module.bias.zero_()

	def forward(self, x):
		out = self.pool(F.relu(self.conv1(x)))
		out = self.pool(F.relu(self.conv2(out)))
		out = self.flat(out)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out
	
	def feature_extractor(self, x):
		out = self.pool(F.relu(self.conv1(x)))
		out = self.pool(F.relu(self.conv2(out)))
		out = self.flat(out)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		return out
    
class MiniAlexNet(nn.Module):
	def __init__(self, n_channels: int = 3, num_classes: int = 10) -> None:
		super().__init__()

		# Padding=1 ensures: 32x32 -> conv(stride=2) => 16x16, pool => 8x8,
		# then conv => 8x8, pool => 4x4, matching 192*4*4.
		self.conv1 = nn.Conv2d(
			in_channels=n_channels,
			out_channels=64,
			kernel_size=3,
			stride=2,
			padding=1,
			bias=True,
		)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(
			in_channels=64,
			out_channels=192,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=True,
		)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.fc1 = nn.Linear(192 * 4 * 4, 1000, bias=True)
		self.fc2 = nn.Linear(1000, num_classes, bias=True)

		self.reset_parameters()

	def reset_parameters(self) -> None:
		"""Initialize trainable weights i.i.d. N(0, 1/sqrt(fan_in))."""
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
				with torch.no_grad():
					module.weight.normal_(mean=0.0, std=1.0 / math.sqrt(fan_in))
					if module.bias is not None:
						module.bias.zero_()
			elif isinstance(module, nn.Linear):
				fan_in = module.in_features
				with torch.no_grad():
					module.weight.normal_(mean=0.0, std=1.0 / math.sqrt(fan_in))
					if module.bias is not None:
						module.bias.zero_()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		return x



                