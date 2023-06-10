from pathlib import Path
from datetime import datetime

import numpy as np
import sklearn
from sklearn.metrics import fbeta_score

import torch
from torch.utils.tensorboard import SummaryWriter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_writer(model_name: str) -> SummaryWriter:
    """Create a SummaryWriter instance saving to a specific log_dir.
    
    This allows us to save metric histories, predictions, etc., to TensorBoard.
    log_dir is formatted as logs/YYYY-MM-DD/model_name.
    
    Args:
      model_name: Name of model.
    
    Returns:
      A SummaryWriter object saving to log_dir.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path("logs") / timestamp / model_name
    print(f"Created SummaryWriter saving to {log_dir}.")
    return SummaryWriter(log_dir=log_dir)
    
    
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
        pred_labels = outputs.detach().to(DEVICE).sigmoid().gt(0.4).int()
        accuracy = (pred_labels == labels).sum().float().div(labels.size(0)).cpu()
        self.fbeta += fbeta_score(labels.view(-1).cpu().numpy(),
                                  pred_labels.view(-1).cpu().numpy(), beta=0.5)
        self.accuracy += accuracy.item()
        self.loss += loss.item()
    def reset(self):
        """Reset values to 0."""
        self.fbeta = 0
        self.loss = 0
        self.accuracy = 0