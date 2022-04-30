import argparse
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import AutoEncoder, Classifier
from run_reconstruction import run_reconstruction
from run_classification import run_classification

# Initialize device either with CUDA or CPU. For this session it does not
# matter if you run the training with your CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Fix seed to be able to reproduce experiments
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Parse parameters for our training
parser = argparse.ArgumentParser()

parser.add_argument("--task", help="either reconstruction or classification", type=str)
parser.add_argument("--log_framework", help="either tensorboard or wandb", type=str)

parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--batch_size", help="batch size", type=int, default=128)
parser.add_argument("--n_epochs", help="training number of epochs", type=int, default=5)
parser.add_argument("--subset_len", help="length of the subsets", type=float, default=5120)
parser.add_argument("--capacity", help="parameter for dimensioning the NN", type=int, default=64)
parser.add_argument("--latent_dims", help="autoencoder bottleneck dimension", type=int, default=64)


args = parser.parse_args()


assert args.task in ['reconstruction', 'classification'], "Task NOT valid. The options are either reconstruction or classification"
assert args.log_framework in ['tensorboard', 'wandb'], "Framework NOT valid. The options are either tensorboard or wandb"

# Load MNIST train and validation sets
mnist_trainset = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
mnist_valset = datasets.MNIST('data', train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

# We don't need the whole dataset, we will pick a subset
train_dataset = Subset(mnist_trainset, list(range(args.subset_len)))
val_dataset = Subset(mnist_valset, list(range(args.subset_len)))


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False)


if args.task == 'reconstruction':
    model = AutoEncoder(
        args.capacity,
        args.latent_dims
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)

    # Put the model into 'device'
    model = model.to(device)

    run_reconstruction(
        args,
        model,
        optimizer,
        train_loader,
        val_loader,
    )
else:

    model = Classifier(
        args.capacity,
        args.latent_dims
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)

    # Put the model into 'device'
    model = model.to(device)

    run_classification(
        args,
        model,
        optimizer,
        train_loader,
        val_loader,
    )
