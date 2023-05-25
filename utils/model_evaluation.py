from pathlib import Path
from typing import Optional, List, Tuple
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
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from .data_preparation import get_rect_dset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_validation_rects(model: nn.Module, 
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