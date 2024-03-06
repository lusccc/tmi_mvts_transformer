import numpy as np
import torch
from logzero import logger

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, monitor_on='loss'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.delta = delta
        self.monitor_on = monitor_on

    def __call__(self, val_metric):
        val_metric = -val_metric if self.monitor_on == 'accuracy' else val_metric

        score = -val_metric

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.warn(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self

