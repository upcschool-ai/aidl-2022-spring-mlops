import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super().__init__()
        self.c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=4,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c * 2,
                               kernel_size=4, stride=2, padding=1)
        self.linear = nn.Linear(in_features=self.c * 2 * 7 * 7,
                                out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super().__init__()
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_dims,
                            out_features=self.c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c * 2, out_channels=self.c,
                                        kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1,
                                        kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c * 2, 7, 7)
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super().__init__()
        self.encoder = Encoder(capacity, latent_dims)
        self.decoder = Decoder(capacity, latent_dims)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self, capacity, latent_dims):
        super().__init__()
        self.encoder = Encoder(capacity, latent_dims)
        self.linear = nn.Linear(latent_dims, 10)

    def forward(self, x):
        x = self.encoder(x)
        return self.linear(x)