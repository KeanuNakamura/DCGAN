import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
import glob
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from PIL import Image
from typing import List,Callable
import torch.optim as optim

device = torch.device('cuda')

file_paths = []
dir_path = 'monet_paintings'
pattern = os.path.join(dir_path,'*.jpg')

for file_path in glob.glob(pattern):
    file_paths.append(file_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class MonetImageDataset(Dataset):
    def __init__(self, file_paths: List[str], transform: Callable):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self,idx):
        file_path = self.file_paths[idx]
        img = Image.open(file_path).convert('RGB')
        img_tensors = self.transform(img)
        return img_tensors

train_dataset = MonetImageDataset(
    file_paths = file_paths,
    transform = transform,
)

train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)

for batch, batch_stuff in enumerate(train_dataloader):
    print(batch_stuff,batch_stuff.shape)
    if batch==0:
        break

#Hyperparameters
latent_dim = 100
g_lr = 0.0006
d_lr = 0.0002
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 100

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
 
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (128, 64, 64)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
 
        self.model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.BatchNorm2d(64, momentum=0.1), 
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128, momentum=0.1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, momentum=0.1),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(278784, 1),
        nn.Sigmoid()
    )
 
    def forward(self, img):
        validity = self.model(img)
        return validity

    def forward(self, img):
        validity = self.model(img)
        return validity

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters()\
                         , lr=g_lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters()\
                         , lr=d_lr, betas=(beta1, beta2))

for i, batch in enumerate(train_dataloader):
    real_images = batch.to(device)

    valid = torch.ones(real_images.size(0), 1, device=device)
    fake = torch.zeros(real_images.size(0), 1, device=device)

    real_images=real_images.to(device)
    optimizer_D.zero_grad()
    z = torch.randn(real_images.size(0), latent_dim, device=device)
    fake_images = generator(z)
    print(discriminator(real_images).shape)
    print(real_images.shape)
    print(z.shape)
    print(fake_images.shape)
    if i==0:
        break

import torchvision

for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        real_images = batch.to(device)

        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)

        real_images = real_images.to(device)

        optimizer_D.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim, device=device)
        fake_images = generator(z)

        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()

        gen_images = generator(z)

        g_loss = adversarial_loss(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]\
                        Batch {i+1}/{len(train_dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(generated,\
                                        nrow=4, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()

with torch.no_grad():
    z = torch.randn(4,latent_dim,device=device)
    generated = generator(z).detach().cpu()
    grid = torchvision.utils.make_grid(generated,\
                                        nrow=2, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.show()

torch.save(generator,'generator_model.pth')

model = torch.load('generator_model.pth')
model.eval()
with torch.no_grad():
    z = torch.randn(4,100,device=device)
    generated = model(z).detach().cpu()
    grid = torchvision.utils.make_grid(generated,\
                                        nrow=2, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.show()