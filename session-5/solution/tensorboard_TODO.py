import os
import torch
import datetime

from Logger import Logger
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger(Logger):

    def __init__(self, task, model):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join("../logs",
                              f"{task}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer = SummaryWriter(log_dir=logdir)

    def log_reconstruction_training(self, model, epoch, train_loss_avg, val_loss_avg, reconstruction_grid):

        # TODO: Log train reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/train_loss" as tag
        self.writer.add_scalar('Reconstruction/train_loss', train_loss_avg, epoch)

        # TODO: Log validation reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/val_loss" as tag
        self.writer.add_scalar('Reconstruction/val_loss', val_loss_avg, epoch)

        # TODO: Log a batch of reconstructed images from the validation set.
        #  Use the reconstruction_grid variable returned above.
        self.writer.add_image('images', reconstruction_grid, epoch)

        # TODO: Log the weights values and grads histograms.
        #  Tip: use f"{name}/value" and f"{name}/grad" as tags
        for name, weight in model.encoder.named_parameters():
            self.writer.add_histogram(f"{name}/value", weight, epoch)
            self.writer.add_histogram(f"{name}/grad", weight.grad, epoch)

    def log_classification_training(self, model, epoch, train_loss_avg,
                                    val_loss_avg, val_acc_avg, train_acc_avg,
                                    fig):
        # TODO: Log confusion matrix figure to tensorboard
        self.writer.add_figure('Confusion Matrix', fig, epoch)

        # TODO: Log validation loss to tensorboard.
        #  Tip: use "Classification/val_loss" as tag
        self.writer.add_scalar('Classification/val_loss', val_loss_avg,
                          epoch)

        # TODO: Log validation accuracy to tensorboard.
        #  Tip: use "Classification/val_acc" as tag
        self.writer.add_scalar('Classification/val_acc', val_acc_avg, epoch)

        # TODO: Log training loss to tensorboard.
        #  Tip: use "Classification/train_loss" as tag
        self.writer.add_scalar('Classification/train_loss', train_loss_avg,
                          epoch)  ## writer...

        # TODO: Log training accuracy to tensorboard.
        #  Tip: use "Classification/train_acc" as tag
        self.writer.add_scalar('Classification/train_acc', train_acc_avg, epoch)

    def log_model_graph(self, model, train_loader):
        """
        TODO:
        We are going to log the graph of the model to Tensorboard. For that, we need to
        provide an instance of the AutoEncoder model and a batch of images, like you'd
        do in a forward pass.
        """

        batch, _ = next(iter(train_loader))
        self.writer.add_graph(model, batch)

    def log_embeddings(self, model, train_loader, device):
        list_latent = []
        list_images = []
        for i in range(10):
            batch, _ = next(iter(train_loader))

            # forward batch through the encoder
            list_latent.append(model.encoder(batch.to(device)))
            list_images.append(batch)

        latent = torch.cat(list_latent)
        images = torch.cat(list_images)

        # TODO: Log latent representations (embeddings) with their corresponding labels (images)
        self.writer.add_embedding(latent, label_img=images)

        # Be patient! Projector logs can take a while

