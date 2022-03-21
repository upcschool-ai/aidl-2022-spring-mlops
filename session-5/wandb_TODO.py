import datetime

import numpy as np
import pandas as pd

import torch
import wandb

from Logger import Logger


class WandbLogger(Logger):

    def __init__(self, task, model):
        wandb.login()
        wandb.init(project="hands-on-monitoring")
        wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

        # TODO: Log weights and gradients to wandb. Doc: https://docs.wandb.ai/ref/python/watch


    def log_reconstruction_training(self, model, epoch, train_loss_avg,
                                    val_loss_avg, reconstruction_grid):

        # TODO: Log train reconstruction loss to wandb


        # TODO: Log validation reconstruction loss to wandb


        # TODO: Log a batch of reconstructed images from the validation set


        pass


    def log_classification_training(self, model, epoch, train_loss_avg,
                                    val_loss_avg, val_acc_avg, train_acc_avg,
                                    fig):
        # TODO: Log confusion matrix figure to wandb


        # TODO: Log validation loss to wandb
        #  Tip: use the tag 'Classification/val_loss'


        # TODO: Log validation accuracy to wandb
        #  Tip: use the tag 'Classification/val_acc'


        # TODO: Log training loss to wandb
        #  Tip: use the tag 'Classification/train_loss'


        # TODO: Log train accuracy to wandb
        #  Tip: use the tag 'Classification/train_acc'


        pass


    def log_embeddings(self, model, train_loader, device):
        out = model.encoder.linear.out_features
        columns = np.arange(out).astype(str).tolist()
        columns.insert(0, "target")
        columns.insert(0, "image")

        list_dfs = []

        for i in range(3): # take only 3 batches of data for plotting
            images, labels = next(iter(train_loader))

            for img, label in zip(images, labels):
                # forward img through the encoder
                image = wandb.Image(img)
                label = label.item()
                latent = model.encoder(img.to(device).unsqueeze(dim=0)).squeeze().detach().cpu().numpy().tolist()
                data = [image, label, *latent]

                df = pd.DataFrame([data], columns=columns)
                list_dfs.append(df)
        embeddings = pd.concat(list_dfs, ignore_index=True)

        # TODO: Log latent representations (embeddings)


    def log_model_graph(self, model, train_loader):
        # Wandb does not support logging the model graph
        pass