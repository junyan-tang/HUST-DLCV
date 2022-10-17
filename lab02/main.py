import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader

image_size = [1, 28, 28]
latent_dim = 96
batch_size = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # use fully-connected layers
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, int(np.prod(image_size))),
            nn.Sigmoid(),
        )

        # use convolution layers
        # self.Linear = nn.Sequential(nn.Linear(latent_dim, 8192))
        # self.model = nn.Sequential(
        #     nn.BatchNorm2d(128),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 1, 3, stride=1, padding=1),
        #     nn.Tanh(),
        # )

    def forward(self, z):
        # shape of z: [batch_size, latent_dim]
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)
        # convolution
        # out = self.Linear(z)
        # out = out.view(out.shape[0], 128, 8, 8)
        # output = self.model(out)
        # image = output.reshape(z.shape[0], *image_size)
        return image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # use fully-connected layers
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_size)), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # use convolution layers
        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout2d(0.25),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout2d(0.25),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout2d(0.25),
        #     nn.Conv2d(64, 256, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout2d(0.25),
        #     nn.BatchNorm2d(256),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, image):
        # shape of image: [batch_size, 1, 28, 28]
        probs = self.model(image.reshape(image.shape[0], -1))
        return probs


# training
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [torchvision.transforms.Resize(28),
                                          torchvision.transforms.ToTensor(),
                                          ]
                                     ))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

generator = Generator()
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)

loss_function = nn.BCELoss()

num_epoch = 100
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        images, _ = mini_batch
        z = torch.randn(batch_size, latent_dim)

        generate_images = generator(z)
        g_optimizer.zero_grad()
        recons_loss = torch.abs(generate_images - images).mean()
        target = torch.ones(batch_size, 1)
        g_loss = recons_loss * 0.05 + loss_function(discriminator(generate_images), target)
        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        real_loss = loss_function(discriminator(images), target)
        fake_loss = loss_function(discriminator(generate_images.detach()), torch.zeros(batch_size, 1))

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        if i % 500 == 0:
            print(f"step:{len(dataloader)*epoch+i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}")
        if i % 800 == 0:
            image = generate_images[:16].data
            torchvision.utils.save_image(image, f"image_{len(dataloader)*epoch+i}.png", nrow=4)


z = torch.rand(2, latent_dim)
a = torch.FloatTensor(latent_dim, 10)
for i in range(latent_dim):
    a[i] = torch.linspace(z[0][i], z[1][i], 10)

b = a.t()
gen_images = generator(b)
torchvision.utils.save_image(gen_images.data[:], "images_trans.png", normalize=True, nrow=10)
