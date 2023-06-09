import gc

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from utils.data_preparation import get_rect_dset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECISION_BOUNDARY = 0.4

def predict_on_test_fragments(model: torch.nn.Module,
                              fragment_path: Path | str,
                              z_start: int,
                              z_dim: int,
                              buffer: int, 
                              batch_size: int,
                              decision_boundary: float = 0.4) -> np.ndarray:
    """Generate predictions on a fragment.
    
    Args:
      model: Model to use for prediction.
      fragment_path: Path to fragment (e.g. '/data/test/a').
      z_start: z-value of lowest fragment layer to use.
      z_dim: Number of layers to use.
      buffer: Radius of subvolumes in x and y direction. Thus, each subvolume
        is an array of shape (z_dim, 2*buffer+1, 2*buffer+1).
      batch_size: Number of subvolumes in one batch.
      decision_boundary: Decision boundary for predictions. If the model
        returns a probability above this number for a given pixel, we classify
        that pixel as containing ink.
    
    Returns:
      Predictions on the fragment, as a boolean NumPy array.
    """
    fragment_path = Path(fragment_path)
    model.eval()
    outputs = []
    test_dset = get_rect_dset(fragment_path, z_start=z_start,
                              z_dim=z_dim, buffer=buffer)
    test_loader = data.DataLoader(test_dset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for i, (subvolumes, _) in enumerate(tqdm(test_loader, desc=f"Predicting on fragment {fragment_path}")):
            output = model(subvolumes.to(DEVICE)).view(-1).sigmoid().cpu().numpy()
            outputs.append(output)
    image_shape = eval_dset.image_stack[0].shape

    pred_image = np.zeros(image_shape, dtype=np.uint8)
    outputs = np.concatenate(outputs)
    for (y, x, _), prob in zip(test_dset.pixels[:outputs.shape[0]], outputs):
        pred_image[y, x] = prob > decision_boundary
    pred_images.append(pred_image)

    print("Finished with fragment", test_fragment)
    return pred_image
                                            
    
def rle(output: np.ndarray) -> str:
    """Turn a NumPy array of booleans to a run-length encoded string."""
    flat_img = np.where(output > 0.4, 1, 0).astype(np.uint8)
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))

                                                
def make_csv(pred_images: list[np.ndarray], save_path: Path | str | None):
    """Save a list of predicted images as a run-length encoded CSV file.
    
    Args:
      pred_images: List of two NumPy arrays of predictions (which we assume
        correspond to test fragments a and b respectively).
      save_path: Path to save CSV file (e.g. '/kaggle/working/submission.csv').
        If None, the file will not be saved.
    Returns:
      A DataFrame containing run-length encodings of the two fragments.
    """
    submission = defaultdict(list)
    for fragment_id, fragment_name in enumerate(['a', 'b']):
        submission["Id"].append(fragment_name)
        submission["Predicted"].append(rle(pred_images[fragment_id]))
    dataframe = pd.DataFrame.from_dict(submission)
    dataframe.to_csv("/kaggle/working/submission.csv", index=False)
    return dataframe