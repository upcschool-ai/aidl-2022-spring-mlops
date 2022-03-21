import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from time import time

from tensorboard_TODO import TensorboardLogger
from wandb_TODO import WandbLogger

# Initialize device either with CUDA or CPU. For this session it does not
# matter if you run the training with your CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def forward_image(model, image_batch):
    image_batch = image_batch.to(device)
    image_batch_recon = model(image_batch)
    return F.mse_loss(image_batch_recon, image_batch), image_batch_recon


def forward_step(model, loader, set, optimizer):
    reconstruction_grid = None
    loss_list = []
    for i, (image_batch, _) in enumerate(loader):
        if set == 'train':
            loss, recon = forward_image(model, image_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, recon = forward_image(model, image_batch)
                if i == 0:
                    # make_grid returns an image tensor from a batch of data (https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid)
                    reconstruction_grid = make_grid(recon)
        loss_list.append(loss.item())
    return loss_list, reconstruction_grid


def run_reconstruction(args, model, optimizer, train_loader, val_loader):

    if args.log_framework == 'tensorboard':
        logger = TensorboardLogger(args.task, model)
    else:
        logger = WandbLogger(args.task, model)

    logger.log_model_graph(model, train_loader)

    ini = time()
    for epoch in range(args.n_epochs):

        """
        We are first running the evaluation step before a single training.
        Can you think of a reason on why we are doing this?
        ...
        In the validation step we log a batch of reconstructed images at the output
        of our AutoEncoder. We want to verify that at the beginning, without any
        tuning of the parameters, our network returns just noise. In our logger
        we'll be able to see how these reconstructions get better over time.
        """
        model.eval()
        val_loss, reconstruction_grid = forward_step(model, val_loader, 'val', optimizer)
        val_loss_avg = np.mean(val_loss)

        model.train()
        train_loss, _ = forward_step(model, train_loader, 'train', optimizer)
        train_loss_avg = np.mean(train_loss)

        logger.log_reconstruction_training(
            model, epoch, train_loss_avg, val_loss_avg, reconstruction_grid
        )

        print(
            f"Epoch [{epoch} / {args.n_epochs}] average reconstruction error: {train_loss_avg}")

    print(f"Training took {round(time() - ini, 2)} seconds")

    logger.log_embeddings(model, train_loader, device)