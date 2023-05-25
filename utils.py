from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_train_and_val_dsets(
    fragments_to_use: List[int],
    data_path: Path | str,
    z_start: int | List[int],
    z_dim: int,
    buffer: int,
    validation_rects: List[Tuple[int]],
    shuffle: bool = True,
    amount_of_data: int | float | None = None,
) -> Tuple[data.Dataset, data.Dataset]:
    """Get training and validation datasets from a list of fragments.
    
    Construct two datasets from the fragments named in fragments_to_use. The
    items in each dataset are cuboidal subvolumes of length 2*buffer+1 in the
    x and y directions and length z_dim in the z direction. A rectangular chunk
    is held out of the training set for each fragment, and used to define the
    validation set. The datasets can optionally be shuffled and restricted in
    size.
    
    Args:
      fragments_to_use: List of fragments to include in the dataset. All
        entries should be in {1, 2, 3}.
      data_path: Path to data folder.
      z_start: Lowest z-value to use from each fragment. Should be between 0
        and 64. If an int, the same z-value is used for all fragments; if a
        list, z_start[i] is used for fragments_to_use[i].
      z_dim: Number of z-values to use from each fragment. This should be the
        same for all fragments so that the model can work with data of constant
        shape.
      buffer: Radius of subvolumes in x and y directions. Thus, each item in
        the dataset will be a subvolume of size 2*buffer+1 x 2*buffer+1 x z_dim.
      validation_rects: Rectangle to hold out from each fragment to form the
        validation set. Each rectangle is a tuple in the format:
          (top_left_corner_x, top_left_corner_y, width, height)
        Use show_labels_with_rects() to double-check the rectangles.
      shuffle: Whether to shuffle the datasets before returning (default True).
      amount_of_data: If None, return the whole dataset. If an int, return a
        training dataset with that many items, and a proportionally long
        validation dataset. If a float, return that fraction of the total
        training and validation datasets (it must be between 0 and 1). You
        should probably use shuffle=True if you want to use this parameter.
    
    Returns:
      A tuple (train_dset, val_dset) of the training and validation datasets.
    
    Example usage:
      train_dset, val_dset = get_train_and_val_dsets(
          fragments_to_use=[1, 2, 3],
          data_path='data',
          z_start=16,
          z_dim=32,
          buffer=30,
          validation_rects=[(1100, 3500, 700, 950), ...],
          shuffle=True,
          amount_of_data: 0.1)
    """
    # This may have a memory leak. It seemed to take more memory when I called
    # it multiple times.
    data_path = Path(data_path)
    train_path = data_path / "train"
    
    if isinstance(z_start, int):
        z_start = [z_start for _ in fragments_to_use]
    
    train_dsets = []
    val_dsets = []
    for i, fragment in enumerate(fragments_to_use):
        # Path for ith fragment
        prefix = train_path / str(fragment)
        
        # read images
        images = [
            np.array(Image.open(filename), dtype=np.float32)/65535.0
            for filename
            in tqdm(
                sorted((prefix / "surface_volume").glob("*.tif"))[z_start[i]:z_start[i]+z_dim],
                desc=f"Loading fragment {fragment}"
            )
        ]
        
        # turn images to tensors
        image_stack = torch.stack([torch.from_numpy(image) for image in images], 
                                  dim=0)

        # get mask and labels
        mask = np.array(Image.open(prefix / "mask.png").convert('1'))
        label = torch.from_numpy(
            np.array(Image.open(prefix / "inklabels.png"))
        ).float()

        # Split our dataset into train and val. The pixels inside the rect are the 
        # val set, and the pixels outside the rect are the train set.
        # Adapted from https://www.kaggle.com/code/jamesdavey/100x-faster-pixel-coordinate-generator-1s-runtime
        
        # Create a Boolean array of the same shape as the bitmask, initially all True
        not_border = np.zeros(mask.shape, dtype=bool)
        not_border[buffer:mask.shape[0]-buffer, buffer:mask.shape[1]-buffer] = True
        arr_mask = mask * not_border
        # define validation rectangle
        rect = validation_rects[i]
        inside_rect = np.zeros(mask.shape, dtype=bool) * arr_mask
        # Sets all indexes with inside_rect array to True
        inside_rect[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = True
        # Set the pixels within the inside_rect to False
        outside_rect = np.ones(mask.shape, dtype=bool) * arr_mask
        outside_rect[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = False
        pixels_inside_rect = torch.tensor(np.argwhere(inside_rect))
        pixels_outside_rect = torch.tensor(np.argwhere(outside_rect))

        # define datasets
        train_dset = SubvolumeDataset(image_stack, label, pixels_outside_rect, buffer, z_dim)
        val_dset = SubvolumeDataset(image_stack, label, pixels_inside_rect, buffer, z_dim)
        train_dsets.append(train_dset)
        val_dsets.append(val_dset)
        
    # Concatenate datasets
    train_dset = data.ConcatDataset(train_dsets)
    val_dset = data.ConcatDataset(val_dsets)
    
    # Find desired lengths of datasets to return
    train_length = len(train_dset)
    val_length = len(val_dset)
    if amount_of_data is None:
        desired_train_length = train_length
        desired_val_length = val_length
    elif isinstance(amount_of_data, int):
        desired_train_length = amount_of_data
        desired_val_length = int(desired_train_length / train_length * val_length)
    elif isinstance(amount_of_data, float):
        desired_train_length = int(amount_of_data * train_length)
        desired_val_length = int(amount_of_data * val_length)
    else:
        raise TypeError(f"Bad type <{type(amount_of_data)}> for amount_of_data: expected None, int, or float")
    
    # Return subsets of appropriate length
    if not shuffle:
        return (data.Subset(train_dset, range(desired_train_length)),
                data.Subset(val_dset, range(desired_val_length)))
    return (data.Subset(train_dset, torch.randperm(train_length)[:desired_train_length]),
            data.Subset(val_dset, torch.randperm(val_length)[:desired_val_length]))
            
            
def get_rect_dset(
    fragment: int,
    data_path: Path | str,
    z_start: int,
    z_dim: int,
    buffer: int,
    rect: Tuple[int],
) -> data.Dataset:
    """Get a dataset consisting of a rectangle from a single fragment.
    
    This returns one unshuffled dataset, which should only be used for
    validation or visualization.
    
    Args:
      fragment: Fragment to get rectangle from. Should be in {1, 2, 3}.
      data_path: Path to data folder.
      z_start: Lowest z-value to use from the fragment. Should be between 0
        and 64. 
      z_dim: Number of z-values to use.
      buffer: Radius of subvolumes in x and y directions. Thus, each item in
        the dataset will be a subvolume of size 2*buffer+1 x 2*buffer+1 x z_dim.
      rect: Rectangle to use from the fragment. Should be a tuple of the form:
          (top_left_corner_x, top_left_corner_y, width, height)
        Use show_labels_with_rects() to double-check the rectangle.
    
    Returns:
      A dataset consisting of subvolumes from the given fragment inside the
      given rectangle.
    """
    data_path = Path(data_path)
    train_path = data_path / "train"
    
    # Path for ith fragment
    prefix = train_path / str(fragment)
        
    # read images
    images = [
        np.array(Image.open(filename), dtype=np.float32)/65535.0
        for filename
        in tqdm(
            sorted((prefix / "surface_volume").glob("*.tif"))[z_start : z_start + z_dim],
            desc=f"Loading fragment {fragment}"
        )
    ]
        
    # turn images to tensors
    image_stack = torch.stack([torch.from_numpy(image) for image in images], 
                              dim=0)

    # get mask and labels
    mask = np.array(Image.open(prefix / "mask.png").convert('1'))
    label = torch.from_numpy(
        np.array(Image.open(prefix / "inklabels.png"))
    ).float()

    # Split our dataset into train and val. The pixels inside the rect are the 
    # val set, and the pixels outside the rect are the train set.
    # Adapted from https://www.kaggle.com/code/jamesdavey/100x-faster-pixel-coordinate-generator-1s-runtime

    # Create a Boolean array of the same shape as the bitmask, initially all True
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[buffer:mask.shape[0]-buffer, buffer:mask.shape[1]-buffer] = True
    arr_mask = mask * not_border
    inside_rect = np.zeros(mask.shape, dtype=bool) * arr_mask
    # Sets all indexes with inside_rect array to True
    inside_rect[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = True
    # Set the pixels within the inside_rect to False
    pixels_inside_rect = torch.tensor(np.argwhere(inside_rect))

    # define dataset
    return SubvolumeDataset(image_stack, label, pixels_inside_rect, buffer, z_dim)
        
    
def show_labels_with_rects(
    fragments_to_use: List[int],
    data_path: Path | str,
    validation_rects: List[Tuple[int]],
    ):
    """Plot the inklabels of given fragments, with validation rects highlighted.
    
    Args:
      fragments_to_use: List of fragments to show. All entries should be in 
        {1, 2, 3}.
      data_path: Path to data folder.
      validation_rect: Rectangle to highlight on each fragment. Each rectangle
        is a tuple in the format:
          (top_left_corner_x, top_left_corner_y, width, height)
    """
    data_path = Path(data_path)
    train_path = data_path / "train"
    
    for i, fragment in enumerate(fragments_to_use):
        plt.subplot(1, len(fragments_to_use), i+1) 
        label = np.array(Image.open(train_path / str(fragment) / "inklabels.png"))
        plt.imshow(label, cmap="gray")
        rect = validation_rects[i]
        patch = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor='r', facecolor='none')
        ax = plt.gca()
        ax.add_patch(patch)
        ax.axis('off')


class SubvolumeDataset(data.Dataset):
    def __init__(self, image_stack, label, pixels, buffer, z_dim):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.buffer = buffer
        self.z_dim = z_dim
    def __len__(self):
        return len(self.pixels)
    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[
            :, y-self.buffer:y+self.buffer+1, x-self.buffer:x+self.buffer+1
        ].view(1, self.z_dim, self.buffer*2+1, self.buffer*2+1)
        inklabel = self.label[y, x].view(1)
        return subvolume, inklabel   
    
    
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
        self.fbeta += fbeta_score(labels.view(-1).cpu().numpy(),
                                  pred_labels.view(-1).cpu().numpy(), beta=0.5)
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
    metrics for visualization or comparison. Optionally writes metric hsitories
    to TensorBoard.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: data.DataLoader,
                 val_loader: Optional[data.DataLoader] = None,
                 optimizer=optim.SGD,
                 criterion=nn.BCEWithLogitsLoss(),
                 lr: float = 0.03,
                 writer: Optional[SummaryWriter] = None,
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
          (3) The use of a learning rate scheduler means that this loop should
            not be called multiple times for the same model. Thus, if we train
            for 30000 epochs and decide after seeing the output that we want to
            train for longer, we would have to train from scratch on a new model.
            This isn't a huge deal, and could be fixed if we can understand how
            to chain together learning rate schedulers.
        
        Args:
          epochs: Number of training batches to use.
          val_epochs: Number of validation batches to use each time validation
            is performed.
          val_period: Number of epochs to train for in between each occurrence
            of validation (default 500).
        """
        # Note, this scheduler should not be used if one plans to call
        # train_eval_loop multiple times.
            
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
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
            scheduler.step()
            # Updates the training_loss, training_fbeta and training_accuracy
            train_metrics.update(outputs, inklabels, loss)
            if (i + 1) % val_period == 0:
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
        
def predict_validation_rects(model, 
                             fragments_to_use: List[int],
                             data_path: Path | str,
                             z_start: int | List[int],
                             z_dim: int,
                             buffer: int,
                             validation_rects: List[Tuple[int]],
                             decision_boundary: float = 0.4,
                             writer: SummaryWriter | None = None):
    """Predict ink labels of given rectangles in the training fragments.
    Display them and add them to TensorBoard.

    Args:
      model: Model to use for prediction.
      fragments_to_use: List of fragments to predict on. All entries should
        be in {1, 2, 3}.
      data_path: Path to data folder.
      z_start: Lowest z-value to use from each fragment. Should be between 0
        and 64. 
      z_dim: Number of z-values to use from each fragment.
      buffer: Radius of subvolumes in x and y directions. Thus, each item in
        the dataset will be a subvolume of size 2*buffer+1 x 2*buffer+1 x z_dim.
      validation_rects: Rectangle to predict on in each fragment. Each
        rectangle is a tuple in the format:
          (top_left_corner_x, top_left_corner_y, width, height)
      decision_boundary: Threshold for predicting a pixel as containing ink
        (default 0.4).
      writer: Optional SummaryWriter object used to add images to TensorBoard.
    """
    
    # clean inputs
    data_path = Path(data_path)
    if isinstance(z_start, int):
        z_start = [z_start for _ in fragments_to_use]
    BATCH_SIZE = 32

    # initialize figure
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle('Predicted')
    axs_pred = subfigs[0].subplots(nrows=1, ncols=len(fragments_to_use))
    subfigs[1].suptitle('Actual')
    axs_actual = subfigs[1].subplots(nrows=1, ncols=3)
    
    model.eval()

    for i, (fragment, rect) in enumerate(zip(fragments_to_use, validation_rects)):
        # define validation dataset
        val_dset = get_rect_dset(fragment, data_path, z_start[i], z_dim,
                                 buffer, rect)
        
        val_loader = data.DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False)
        
        # predict on validation dataset
        outputs = []
        with torch.no_grad():
            for subvolumes, _ in tqdm(val_loader,
                                      desc=f"Predicting on fragment {fragment}"):
                output = model(subvolumes.to(DEVICE)).view(-1).sigmoid().cpu().numpy()
                outputs.append(output)
        image_shape = val_dset.image_stack[0].shape

        # get boolean predictions using decision boundary
        pred_image = np.zeros(image_shape, dtype=np.uint8)
        outputs = np.concatenate(outputs)
        for (y, x), prob in zip(val_dset.pixels[:outputs.shape[0]], outputs):
            pred_image[y, x] = prob > decision_boundary
            
        # clean up
        del val_dset
        del val_loader
        gc.collect()

        # plot predictions
        axs_pred[i].imshow(pred_image, cmap='gray')
        axs_pred[i].set_xlim([rect[0], rect[0]+rect[2]])
        axs_pred[i].set_ylim([rect[1]+rect[3], rect[1]])
        axs_pred[i].axis('off')

        # plot actual labels
        label = np.array(Image.open(data_path / "train" / f"{i+1}/inklabels.png"))
        axs_actual[i].imshow(label, cmap='gray')
        axs_actual[i].set_xlim([rect[0], rect[0]+rect[2]])
        axs_actual[i].set_ylim([rect[1]+rect[3], rect[1]])
        axs_actual[i].axis('off')
        
    # log to tensorboard
    if writer is not None:
        writer.add_figure('Validation predictions', fig)
    
    return fig
