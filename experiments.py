import torch
import torchvision
from gan import Generator, Discriminator, generator_loss, discriminator_loss
import matplotlib.pyplot as plt

epochs = 50
num_examples_to_generate = 16
batch_size = 256
buffer_size = 6000
noise_length = 100
lr = 1e-4
beta1 = .5

transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
dataset = torchvision.datasets.MNIST("mnist", download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator()
discriminator = Discriminator()

gen_opt = torch.optim.Adam(generator.parameters(), lr, betas=(beta1, .99))
dis_opt = torch.optim.Adam(discriminator.parameters(), lr, betas=(beta1, .99))

def train_step(images):
	noise = torch.randn([batch_size, noise_length])
	fake_images = generator(noise)

	real_out = discriminator(images)
	fake_out = discriminator(fake_images)

	gen_loss = generator_loss(fake_out)
	gen_loss.backward(retain_graph=True)
	gen_opt.step()

	dis_loss = discriminator_loss(real_out, fake_out)
	dis_loss = dis_loss.backward()
	dis_opt.step()

	return fake_images

def train():
	for epoch in range(epochs):
		print(epoch)
		i = 0
		for images, _ in dataloader:
			last_generated = train_step(images)
			plt.imshow(last_generated.detach()[0, 0, :, :])
			plt.savefig('generated_images/gan{}.png'.format(i))
			i += 1

train()
generate_images()
