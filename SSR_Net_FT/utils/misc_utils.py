import torch

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait after last time validation loss improved.
                                Defaults to 10.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.counter < self.patience:
            self.counter += 1
        else:
            self.early_stop = True
        return self.early_stop

