from pathlib import Path
from typing import Optional, List, Tuple
import gc

from PIL import Image
# disable PIL.DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.utils.data as data

from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SubvolumeDataset(data.Dataset):
    """Dataset containing cubical subvolumes of image stack."""
    def __init__(self, image_stack: torch.Tensor, label: torch.Tensor | None,
                 pixels: torch.Tensor, buffer: int):
        """Create a new SubvolumeDataset.
        
        Args:
          image_stack: 3D image data, as a tensor of voxel values of shape
            (z_dim, y_dim, x_dim).
          label: ink labels for the image stack. A tensor of shape (y_dim, x_dim).
            For testing data, instead pass label=None.
          pixels: Tensor listing pixels to use as centers of subvolumes, of
            shape (num_pixels, 2). Each row of pixels gives coordinates (y, x) of
            a single subvolume center.
          buffer: radius of each subvolume in the x and y direction. Thus, each
            subvolume has shape (z_dim, 2*buffer+1, 2*buffer+1).
        """
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.buffer = buffer
        
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.pixels)
    
    def __getitem__(self, index):
        """Get a subvolume from the dataset.
        
        If the dataset was defined without label data, the returned label will
        be -1.
        """
        y, x = self.pixels[index]
        subvolume = self.image_stack[
            :, y-self.buffer:y+self.buffer+1, x-self.buffer:x+self.buffer+1
        ].view(1, -1, self.buffer*2+1, self.buffer*2+1)
        if self.label is None:
            inklabel = -1
        else:
            inklabel = self.label[y, x].view(1)
        return subvolume, inklabel   
    
    
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
        train_dset = SubvolumeDataset(image_stack, label, pixels_outside_rect, buffer)
        val_dset = SubvolumeDataset(image_stack, label, pixels_inside_rect, buffer)
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
    fragment: int | str,
    data_path: Path | str,
    z_start: int,
    z_dim: int,
    buffer: int,
    rect: Tuple[int] | None = None,
) -> data.Dataset:
    """Get a dataset consisting of a rectangle from a single fragment.
    
    This returns one unshuffled dataset, which should only be used for
    validation, visualization, or testing.
    
    Args:
      fragment: Fragment to get rectangle from. Should be one of 1, 2, 3, 'a',
        'b', where 1, 2, 3 are training fragments and 'a', 'b' are testing
        fragments.
      data_path: Path to data folder.
      z_start: Lowest z-value to use from the fragment. Should be between 0
        and 64. 
      z_dim: Number of z-values to use.
      buffer: Radius of subvolumes in x and y directions. Thus, each item in
        the dataset will be a subvolume of size 2*buffer+1 x 2*buffer+1 x z_dim.
      rect: Rectangle to use from the fragment. Should be a tuple of the form:
          (top_left_corner_x, top_left_corner_y, width, height)
        Use show_labels_with_rects() to double-check the rectangle. If rect is
        None, the whole dataset will be used.
    
    Returns:
      A dataset consisting of subvolumes from the given fragment inside the
      given rectangle.
    """
    # clean input
    data_path = Path(data_path)
    fragment = str(fragment)
    
    # check if training or testing
    if fragment in {'1', '2', '3'}:
        prefix = data_path / "train" / fragment
    elif fragment in {'a', 'b'}:
        prefix = data_path / "test" / fragment
    else:
        raise ValueError(f"Unrecognized fragment {fragment}.")
        
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
    if fragment in {'1', '2', '3'}:
        label = torch.from_numpy(
            np.array(Image.open(prefix / "inklabels.png"))
        ).float()
    else:
        label = None

    # Create a Boolean array of the same shape as the bitmask, initially all True
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[buffer:mask.shape[0]-buffer, buffer:mask.shape[1]-buffer] = True
    arr_mask = mask * not_border
    if rect is not None:
        inside_rect = np.zeros(mask.shape, dtype=bool) * arr_mask
        # Sets all indexes with inside_rect array to True
        inside_rect[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = True
        # Set the pixels within the inside_rect to False
        pixels = torch.tensor(np.argwhere(inside_rect))
    else:
        pixels = torch.tensor(np.argwhere(arr_mask))

    # define dataset
    return SubvolumeDataset(image_stack, label, pixels, buffer)
        
    
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

