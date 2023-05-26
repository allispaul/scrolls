from pathlib import Path
from typing import Optional, List, Tuple, Callable
import gc

from PIL import Image
# disable PIL.DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import fbeta_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from .logging import MetricsRecorder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer():
    """Bundles together a model, a training and optionally a validation dataset,
    an optimizer, and loss/accuracy/fbeta@0.5 metrics. Stores histories of the
    metrics for visualization or comparison. Optionally writes metric hsitories
    to TensorBoard.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: data.DataLoader,
                 val_loader: data.DataLoader,
                 *,
                 optimizer: type[torch.optim.Optimizer] = optim.SGD,
                 criterion: nn.Module = nn.BCEWithLogitsLoss(),
                 lr: float = 0.03,
                 scheduler: Optional[type[torch.optim.lr_scheduler]] = None,
                 writer: Optional[SummaryWriter] = None,
                 ):
        """Create a Trainer.
        
        Args:
          model: Model to train.
          train_loader: DataLoader containing training data.
          val_loader: DataLoader containing validation data.
          optimizer: Optimizer to use during training; give it a class, not an
            instance (SGD, not SGD()). Default torch.optim.SGD.
          criterion: Loss criterion to minimize. Default nn.BCEWithLogitsLoss().
          lr: Learning rate. Default 0.03.
          scheduler: Optional learning rate scheduler. As with optimizer, give
            a class, which will be initialized at the start of training. If no
            scheduler is given, a constant learning rate is used.
          writer: SummaryWriter object to log performance to TensorBoard. You
            can create this using .logging.create_writer().
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optimizer(model.parameters(), lr=self.lr)
        self.scheduler = scheduler
        self.histories = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'train_fbeta': [],
            'val_loss': [],
            'val_acc': [],
            'val_fbeta': []
        }
        self.writer = writer

    def train_eval_loop(self, epochs, val_epochs, val_period=500):
        """Train model for a given number of epochs, performing validation
        periodically.
        
        Train the model on a number of training batches given by epochs. Every
        val_period training batches, pause training and perform validation on
        val_epochs batches from the validation set. Each time validation is
        performed, the model's loss, accuracy, and F0.5 scores are saved to the
        trainer, and optionally written to TensorBoard.
        
        A few things in this loop we may want to change eventually:
          (1) We assume that epochs is less than the length of train_loader.
          (2) We reinitialize the val_loader iterator every time we perform
            validation. This makes sense if we're shuffling val_loader, but
            for performance reasons we may decide to stop doing this.
        
        Args:
          epochs: Number of training batches to use.
          val_epochs: Number of validation batches to use each time validation
            is performed.
          val_period: Number of epochs to train for in between each occurrence
            of validation (default 500).
        """
        # Note, this scheduler should not be used if one plans to call
        # train_eval_loop multiple times.
            
        if self.scheduler is not None:
            scheduler = self.scheduler(
                self.optimizer, max_lr=self.lr, total_steps=epochs
            )
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=epochs, desc="Training")
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
            if self.scheduler is not None:
                scheduler.step()
            # Updates the training_loss, training_fbeta and training_accuracy
            train_metrics.update(outputs, inklabels, loss)
            if (i + 1) % val_period == 0:
                self.histories['epochs'].append(i+1)
                self.histories['train_loss'].append(train_metrics.loss / val_period)
                self.histories['train_acc'].append(train_metrics.accuracy / val_period)
                self.histories['train_fbeta'].append(train_metrics.fbeta / val_period)
                train_metrics.reset()

                self.model.eval()
                for j, (val_subvolumes, val_inklabels) in enumerate(self.val_loader):
                    if j >= val_epochs:
                        break
                    with torch.inference_mode():
                        val_outputs = self.model(val_subvolumes.to(DEVICE))
                        val_loss = self.criterion(val_outputs, val_inklabels.to(DEVICE))
                    val_metrics.update(val_outputs, val_inklabels, val_loss)
                self.histories['val_loss'].append(val_metrics.loss / j)
                self.histories['val_acc'].append(val_metrics.accuracy / j)
                self.histories['val_fbeta'].append(val_metrics.fbeta / j)
                val_metrics.reset()
                
                # If logging to TensorBoard, add metrics to writer
                if self.writer is not None:
                    self.writer.add_scalars(main_tag="Loss",
                                            tag_scalar_dict={
                                                "train_loss": self.histories['train_loss'][-1],
                                                "val_loss": self.histories['val_loss'][-1],
                                            },
                                            global_step=i)
                    self.writer.add_scalars(main_tag="Accuracy",
                                            tag_scalar_dict={
                                                "train_acc": self.histories['train_acc'][-1],
                                                "val_acc": self.histories['val_acc'][-1],
                                            },
                                            global_step=i)
                    self.writer.add_scalars(main_tag="Fbeta@0.5",
                                            tag_scalar_dict={
                                                "train_fbeta": self.histories['train_fbeta'][-1],
                                                "val_fbeta": self.histories['val_fbeta'][-1],
                                            },
                                            global_step=i)
                    # write to disk
                    self.writer.flush()

    def plot_metrics(self):
        plt.subplot(131)
        plt.plot(self.histories['epochs'], self.histories['train_loss'], label="training")
        plt.plot(self.histories['epochs'], self.histories['val_loss'], label="validation")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss")
        plt.legend()

        plt.subplot(132)
        plt.plot(self.histories['epochs'], self.histories['train_acc'], label="training")
        plt.plot(self.histories['epochs'], self.histories['val_acc'], label="validation")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(133)
        plt.plot(self.histories['epochs'], self.histories['train_fbeta'], label="training")
        plt.plot(self.histories['epochs'], self.histories['val_fbeta'], label="validation")
        plt.xlabel('epoch')
        plt.ylabel('fbeta')
        plt.title("F-Beta")
        plt.legend()
        
        plt.tight_layout()