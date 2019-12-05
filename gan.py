import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, noise_length=100):
		super().__init__()
		self.length1 = 7
		self.fc = nn.Linear(noise_length, 
			self.length1 * self.length1 * 256, 
			bias=False)
		self.bn0 = nn.BatchNorm2d(256)
		self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)
		self.bn1 = nn.BatchNorm2d(128)
		self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=2)

	def forward(self, x):
		x = self.fc(x)

		x = x.view(-1, 256, self.length1, self.length1)
		x = self.bn0(x)
		x = F.leaky_relu(x)

		# TODO: Check that edges aren't lost.
		x = self.deconv1(x)
		x = self.bn1(x)
		x = F.leaky_relu(x)

		x = self.deconv2(x)
		x = self.bn2(x)
		x = F.leaky_relu(x)

		x = self.deconv3(x)
		x = F.sigmoid(x)
		return x


def generator_loss(fake_dis_out):
	return F.binary_cross_entropy(fake_dis_out, torch.ones_like(fake_dis_out))


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2)
		self.dropout1 = nn.Dropout(0.3)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
		self.dropout2 = nn.Dropout(0.3)
		self.fc = nn.Linear(128 * 4 * 4, 1)

	def forward(self, x):
		x = self.conv1(x)
		x = F.leaky_relu(x)
		x = self.dropout1(x)

		x = self.conv2(x)
		x = F.leaky_relu(x)
		x = self.dropout2(x)
		x = x.view(-1, 128 * 4 * 4)
		x = self.fc(x)
		x = F.sigmoid(x)
		return x


def discriminator_loss(fake_dis_out, real_dis_out):
	real_loss =  F.binary_cross_entropy(real_dis_out, torch.zeros_like(real_dis_out))
	fake_loss =  F.binary_cross_entropy(fake_dis_out, torch.zeros_like(fake_dis_out))
	return fake_loss + real_loss