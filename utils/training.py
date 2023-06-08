from pathlib import Path
from typing import Optional, List, Tuple, Callable
from itertools import cycle
import gc
import time

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

from .logging import MetricsRecorder, create_writer

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
                 writer: SummaryWriter | "auto" | None = None,
                 model_name: str | None = None,
                 **kwargs,
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
            can create this using .logging.create_writer(). If writer is
            "auto", a SummaryWriter will automatically be created, using
            model_name (which is required in this case).
          model_name: Name of the model. Will be used to save checkpoints for
            the model and to automatically create a SummaryWriter.
          Keyword arguments prefixed by `optimizer_` or `scheduler_` are passed
            to the optimizer or scheduler, respectively.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = criterion
        self.lr = lr
        self.scheduler_class = scheduler
        self.optimizer_kwargs = dict()
        self.scheduler_kwargs = dict()
        self.scheduler_kwargs.setdefault('max_lr', self.lr)
        for key in kwargs.keys():
            # strip off optimizer or scheduler prefix and add to relevant dict
            if key.startswith('optimizer_'):
                self.optimizer_kwargs.update({key[10:]: kwargs[key]})
            if key.startswith('scheduler_'):
                self.scheduler_kwargs.update({key[10:]: kwargs[key]})
        self.optimizer = optimizer(model.parameters(), lr=self.lr,
                                   **self.optimizer_kwargs)
        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(
                self.optimizer,
                **self.scheduler_kwargs
            )
            
        self.histories = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'train_fbeta': [],
            'val_loss': [],
            'val_acc': [],
            'val_fbeta': [],
            'lr': []
        }
        
        self.model_name = model_name
        self.writer = writer
        if self.writer == "auto":
            if model_name is None:
                raise ValueError('model_name is required with writer="auto"')
            self.writer = create_writer(model_name)
    
    def train_step(self, subvolumes, inklabels):
        self.optimizer.zero_grad()
        outputs = self.model(subvolumes.to(DEVICE))
        loss = self.criterion(outputs, inklabels.to(DEVICE))
        loss.backward()
        self.optimizer.step()
        if self.scheduler_class is not None:
            self.scheduler.step()
        return outputs, loss
            
    def val_step(self, subvolumes, inklabels):
        with torch.inference_mode():
            val_outputs = self.model(subvolumes.to(DEVICE))
            val_loss = self.criterion(val_outputs, inklabels.to(DEVICE))
        return val_outputs, val_loss
    
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
            
        self.model.train()
        
        train_metrics = MetricsRecorder()
        val_metrics = MetricsRecorder()
        
        # It doesn't make sense to have more validation steps than batches in
        # the validation set
        val_epochs = min(val_epochs, len(self.val_loader))
        # estimate total epochs
        total_epochs = epochs + ((epochs // val_period) * val_epochs)
        pbar = tqdm(total=total_epochs, desc="Training")
        self.model.train()
        for i, (subvolumes, inklabels) in enumerate(cycle(self.train_loader)):
            pbar.update()
            if i >= epochs:
                break
            outputs, loss = self.train_step(subvolumes, inklabels)
            # Updates the training_loss, training_fbeta and training_accuracy
            train_metrics.update(outputs, inklabels, loss)
            if (i + 1) % val_period == 0 or i+1 == len(self.train_loader):
                # record number of epochs and training metrics
                self.histories['epochs'].append(i+1)
                self.histories['train_loss'].append(train_metrics.loss / val_period)
                self.histories['train_acc'].append(train_metrics.accuracy / val_period)
                self.histories['train_fbeta'].append(train_metrics.fbeta / val_period)
                train_metrics.reset()

                # record learning rate
                if self.scheduler_class is not None:
                    self.histories['lr'].append(self.scheduler.get_last_lr()[0])
                else:
                    self.histories['lr'].append(self.optimizer.param_groups[0]['lr'])

                # predict on validation data and record metrics
                self.model.eval()
                for j, (val_subvolumes, val_inklabels) in enumerate(self.val_loader):
                    pbar.update()
                    if j >= val_epochs:
                        break
                    val_outputs, val_loss = self.val_step(val_subvolumes, val_inklabels)
                    val_metrics.update(val_outputs, val_inklabels, val_loss)
                self.model.train()
                
                self.histories['val_loss'].append(val_metrics.loss / j)
                self.histories['val_acc'].append(val_metrics.accuracy / j)
                self.histories['val_fbeta'].append(val_metrics.fbeta / j)
                val_metrics.reset()

                # If logging to TensorBoard, add metrics to writer
                if self.writer is not None:
                    self.writer.add_scalars(
                        main_tag="Loss",
                        tag_scalar_dict={
                            "train_loss": self.histories['train_loss'][-1], 
                            "val_loss": self.histories['val_loss'][-1], 
                        }, 
                        global_step=i+1)
                    self.writer.add_scalars(
                        main_tag="Accuracy",  
                        tag_scalar_dict={  
                            "train_acc": self.histories['train_acc'][-1], 
                            "val_acc": self.histories['val_acc'][-1], 
                        }, 
                        global_step=i+1)
                    self.writer.add_scalars(
                        main_tag="Fbeta@0.5", 
                        tag_scalar_dict={ 
                            "train_fbeta": self.histories['train_fbeta'][-1], 
                            "val_fbeta": self.histories['val_fbeta'][-1], 
                        }, 
                        global_step=i+1)
                    self.writer.add_scalar(
                        "Learning rate",
                        self.histories['lr'][-1], 
                        global_step=i+1)
                    # write to disk
                    self.writer.flush()
                    
                # If loss is NaN, the model died and we might as well stop training.
                if np.isnan(self.histories['val_loss'][-1]) or np.isnan(self.histories['train_loss'][-1]):
                    print (f"Model died at training epoch {i+1}, stopping training.")
                    break
    
    def time_train_step(self, n=500):
        """Time training on a given number of training batches. Note that this
        does train the model.
        """
        tic = time.perf_counter()
        self.model.train()
        for i in range(n):
            subvolumes, inklabels = next(iter(self.train_loader))
            self.train_step(subvolumes, inklabels)
        toc = time.perf_counter()
        delta = toc - tic
        print(f"Trained on {n} batches in {delta:.2f}s.")
        return n
        
    def time_val_step(self, n=500):
        """Time prediction on a given number of validation batches."""
        tic = time.perf_counter()
        self.model.eval()
        for i in range(n):
            subvolumes, inklabels = next(iter(self.val_loader))
            self.val_step(subvolumes, inklabels)
        toc = time.perf_counter()
        delta = toc - tic
        print(f"Predicted on {n} validation batches in {delta:.2f}s.")
        return n

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
        
    def save_checkpoint(self, extra):
        
        