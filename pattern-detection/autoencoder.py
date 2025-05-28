import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*3, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64*64*3),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, 64, 64))
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def train_autoencoder(image_paths, epochs=20):
    model = VAE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    data = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        data.append(transform(img))
    data = torch.stack(data).cuda()
    for epoch in range(epochs):
        recon, mu, logvar = model(data)
        loss = ((recon - data) ** 2).mean() + (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def extract_latent_features(model, image_paths):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    data = [transform(Image.open(p).convert('RGB')) for p in image_paths]
    data = torch.stack(data).cuda()
    with torch.no_grad():
        h = model.encoder(data)
        mu, _ = h.chunk(2, dim=1)
    return mu.cpu().numpy()
