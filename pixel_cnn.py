import torch
import torch.nn as nn
import torch.nn.functional as F

def ResidualBlock(nn.Module):

	# 'channels' corresponds to parameter 'h' in van den Oord et al. 2016.
	def __init__(self, channels=16):
		
		super().__init__()

		self.conv1 = nn.Conv2d(2 * channels, channels, kernal_size=1, padding=0)
		self.conv2 = nn.Conv2d(channels, channels, kernal_size=3, padding=1)
		self.conv3 = nn.Conv2d(channels, 2 * channels, kernal_size=1, padding=0)

	def forward(self, x):
		x2 = F.relu(x2)
		x2 = self.conv1(x)
		x2 = F.relu(x2)
		x2 = self.conv2(x2)
		x2 = F.relu(x2)
		x2 = self.conv3(x2)
		return x + x2


def PixelCNN(nn.Module):
	def __init__(self, depth=8, channels=16):
		super().__init__()
		layers = [nn.ResidualBlock(channels=channels) for _ in range(depth)]
		self.convs = nn.Sequential(*layers)
		self.linear = nn.linear(_ , 256)

	def forward(self, x):
		x = self.convs(x)
		x = self.view(-1, _)
		x = self.linear(x)
		return x