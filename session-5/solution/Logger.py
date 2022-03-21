class Logger():

    def log_reconstruction_training(self, model, epoch, train_loss_avg,
                                    val_loss_avg, reconstruction_grid):
        raise NotImplementedError

    def log_classification_training(self, model, epoch, train_loss_avg,
                                    val_loss_avg, val_acc_avg, train_acc_avg,
                                    fig):
        raise NotImplementedError

    def log_model_graph(self, model, train_loader):
        raise NotImplementedError

    def log_embeddings(self, model, train_loader, device):
        raise NotImplementedError
