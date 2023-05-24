import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# renamed from "visualize"
class MetricsRecorder():
    def __init__(self):
        """
        In here we initialize the values to 0
        """
        self.fbeta=0
        self.loss=0
        self.accuracy=0
    def update(self, outputs, labels, loss):
        """
        Takes outputs, labels and loss as input and updates the instance variables fbeta, accuracy and loss
        """
        labels = labels.to(DEVICE)
        pred_labels = outputs.detach().sigmoid().gt(0.4).int()
        accuracy = (pred_labels == labels).sum().float().div(labels.size(0)).cpu()
        self.fbeta += fbeta_score(labels.view(-1).cpu().numpy(), pred_labels.view(-1).cpu().numpy(), beta=0.5)
        self.accuracy += accuracy.item()
        self.loss += loss.item()
    def reset(self):
        """Reset values to 0."""
        self.fbeta = 0
        self.loss = 0
        self.accuracy = 0

class Trainer():
    """Bundles together a model, a training and optionally a validation dataset,
    an optimizer, and loss/accuracy/fbeta@0.5 metrics. Stores histories of the
    metrics for visualization or comparison.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: data.DataLoader,
                 val_loader: Optional[data.DataLoader] = None,
                 optimizer=optim.SGD,
                 criterion=nn.BCEWithLogitsLoss(),
                 lr: float = 0.03,
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optimizer(model.parameters(), lr=self.lr)
        self.histories = {
            'train_loss': [],
            'train_acc': [],
            'train_fbeta': [],
            'val_loss': [],
            'val_acc': [],
            'val_fbeta': []
        }

    def train_eval_loop(self, epochs, validation_epochs):
        # Note, this scheduler should not be used if one plans to call
        # train_eval_loop multiple times.
            
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, total_steps=epochs
        )
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=epochs)
        train_metrics = MetricsRecorder()
        val_metrics = MetricsRecorder()
        for i, (subvolumes, inklabels) in pbar:
            if i >= epochs:
                break
            self.optimizer.zero_grad()
            outputs = self.model(subvolumes.to(DEVICE))
            loss = self.criterion(outputs, inklabels.to(DEVICE))
            loss.backward()
            self.optimizer.step()
            scheduler.step()
            # Updates the training_loss, training_fbeta and training_accuracy
            train_metrics.update(outputs, inklabels, loss)
            if (i + 1) % 500 == 0:
                self.histories['train_loss'].append(train_metrics.loss / 500)
                self.histories['train_acc'].append(train_metrics.accuracy / 500)
                self.histories['train_fbeta'].append(train_metrics.fbeta / 500)
                train_metrics.reset()

                self.model.eval()
                for j, (val_subvolumes, val_inklabels) in enumerate(self.val_loader):
                    if j >= validation_epochs:
                        break
                    with torch.inference_mode():
                        val_outputs = self.model(val_subvolumes.to(DEVICE))
                        val_loss = self.criterion(val_outputs, val_inklabels.to(DEVICE))
                    val_metrics.update(val_outputs, val_inklabels, val_loss)
                self.histories['val_loss'].append(val_metrics.loss / j)
                self.histories['val_acc'].append(val_metrics.accuracy / j)
                self.histories['val_fbeta'].append(val_metrics.fbeta / j)
                val_metrics.reset()

    def plot_metrics(self):
        plt.subplot(131)
        plt.plot(self.histories['train_loss'], label="training")
        plt.plot(self.histories['val_loss'], label="validation")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss")
        plt.legend()

        plt.subplot(132)
        plt.plot(self.histories['train_acc'], label="training")
        plt.plot(self.histories['val_acc'], label="validation")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(133)
        plt.plot(self.histories['train_fbeta'], label="training")
        plt.plot(self.histories['val_fbeta'], label="validation")
        plt.xlabel('epoch')
        plt.ylabel('fbeta')
        plt.title("F-Beta")
        plt.legend()
        
        plt.tight_layout()
