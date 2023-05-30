import os
from pathlib import Path
import shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np
from tqdm import tqdm

def main():
    """Cut training fragments from the Kaggle data directory into pieces.

    You should change DATA_PATH and SAVE_FOLDER to the paths to your data and
    wherever you want the pieces. Each piece is identified by a pair x,y, and
    saved inside a folder with that name. Running this script will create a
    folder structure of the form:
        pieces/train/1/3,0/surface_volume/00.tif,
        pieces/train/1/3,0/surface_volume/01.tif,
        ...,
        pieces/train/1/3,0/ir.png,
        pieces/train/1/3,0/inklabels.png,
        pieces/train/1/3,0/mask.png,
        pieces/train/1/pairs.txt,
        pieces/train/1/4,0/surface_volume/00.tif,
        ...,
    Thus, the directory structure within a given x,y mimics that of the Kaggle
    data. The file pairs.txt lists the pairs x,y that are used. The
    All .tif files are roughly 1000x1000 pixels. I left out the test fragments,
    since they're just pieces of fragment 1.

    NOTE:
      1. Running this script will create about 100GB of data.
      2. I think the PNGs are being saved with the wrong "color mode", which
         means that they look totally black when viewed on an image viewer.
         The data is still there and can be seen if you load the images in
         Python and display them with pyplot, etc.
    """
    DATA_PATH = Path('data/')
    SAVE_FOLDER = DATA_PATH / "pieces"
    FRAGMENT_CUTS = {
        'train/1': (6, 8),
        'train/2': (7, 10),
        'train/3': (5, 7),
        # 'test/a': (6, 3),
        # 'test/b': (6, 5),
    }
    # create save folder if it doesn't exist
    SAVE_FOLDER.mkdir(exist_ok=True)
    for fragment in FRAGMENT_CUTS.keys():
        print(f"Starting on fragment {fragment}")
        fragment_save_folder = SAVE_FOLDER / fragment
        fragment_save_folder.mkdir(parents=True, exist_ok=True)

        # dissect mask
        mask_path = DATA_PATH / fragment / "mask.png"
        x_cuts, y_cuts = FRAGMENT_CUTS[fragment]
        print(f"Dissecting {mask_path}...")
        indices = dissect(mask_path, fragment_save_folder, x_cuts, y_cuts)

        # dissect volume data
        volume_folder = DATA_PATH / fragment / "surface_volume"
        for layer in tqdm(range(64), desc="Dissecting layers"):
            layer_path = volume_folder / f"{layer:02}.tif"
            dissect(layer_path, fragment_save_folder,
                    x_cuts, y_cuts, inside_save_folder="surface_volume")

        # dissect inklabels and IR photo if they exist
        label_path = DATA_PATH / fragment / "inklabels.png"
        if os.path.exists(label_path):
            print(f"Dissecting {label_path}...")
            dissect(label_path, fragment_save_folder, x_cuts, y_cuts)
        ir_path = DATA_PATH / fragment / "ir.png"
        if os.path.exists(ir_path):
            print(f"Dissecting {ir_path}...")
            dissect(ir_path, fragment_save_folder, x_cuts, y_cuts)

        # Remove folders for which the surface volume data is blank
        for  x in range(x_cuts):
            for y in range(y_cuts):
                if (x, y) not in indices:
                    shutil.rmtree(fragment_save_folder / f"{x},{y}")

        # Write list of indices to file
        with open(fragment_save_folder / "pairs.txt", "w") as pairs:
            for (x, y) in indices:
                pairs.write(f"{x},{y}\n")


def dissect(img_path: Path | str, save_folder: Path | str,
            x_cuts: int, y_cuts: int,
            inside_save_folder: Path | str | None = None,
            staggered: bool = True,
            verbose: bool = False) -> list[tuple[int, int]]:
    """Cut grayscale image into rectangular pieces and save them to disk.

    Pieces are saved to:
      {save_folder}/{x_index},{y_index}/{inside_save_folder}/{save_filename}.

    Args:
      img_path: Path to read the image from.
      save_folder: Folder to save pieces to.
      inside_save_folder: Optional folder to put inside the "{x_index},{y_index}"
        folders created.
      x_cuts: Number of cuts to make in the x direction (parallel to the y-axis).
      y_cuts: Number of cuts to make in the y direction (parallel to the x-axis).
      staggered: If True (the default), will step by half the image size in each
        direction before forming the next image. As a result, roughly 2*x_cuts*y_cuts
        images will be created. The point is to ensure that no pixel only occurs on
        the edge of one of the created images. If staggered is False, roughly
        x_cuts*y_cuts images will be created.
      verbose: Whether to print file operations.

    Returns: List of non-blank (x, y) index pairs.
    """
    img_path = Path(img_path)
    save_folder = Path(save_folder)
    if inside_save_folder is None:
        inside_save_folder = Path(".")
    if isinstance(inside_save_folder, str):
        inside_save_folder = Path(inside_save_folder)
    # create save folder if it doesn't exist
    save_folder.mkdir(exist_ok=True)

    img_asarray = np.array(Image.open(img_path))
    y_max, x_max = img_asarray.shape

    # calculate sizes of pieces and amount to step
    y_width = y_max // y_cuts
    x_width = x_max // x_cuts
    y_step = y_width // (1 + staggered)
    x_step = x_width // (1 + staggered)

    indices = []

    for y_idx in range(y_cuts * (1 + staggered)):
        for x_idx in range(x_cuts * (1 + staggered)):
            piece = img_asarray[ y_idx*y_step : y_idx*y_step + y_width, x_idx*x_step:x_idx*x_step + x_width ]
            # if this piece is non-blank, retain its index
            if piece.max() > 0.0:
                indices.append((x_idx, y_idx))
            piece_savepath = (save_folder / f"{x_idx},{y_idx}"
                              / inside_save_folder / img_path.name)
            piece_savepath.parent.mkdir(exist_ok=True)
            if verbose:
                print(f"Saving to {piece_savepath}...")
            Image.fromarray(piece).save(piece_savepath)
    if verbose:
        print("File operations complete.")

    return indices


if __name__ == "__main__":
    main()
