import torch, math
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Optional
from cs336_basics.utils.nn import softmax
from typing import Dict, Optional, Tuple, Iterable, List

def make_batch(dataset, batch_starts, context_length, device):
    x_list = [dataset[batch_start:batch_start+context_length] for batch_start in batch_starts] 
    y_list = [dataset[batch_start+1:batch_start+context_length+1] for batch_start in batch_starts] 
    x = torch.stack([torch.from_numpy(np.copy(x_i).astype(np.float32)) for x_i in x_list], dim=0)
    y = torch.stack([torch.from_numpy(np.copy(y_i).astype(np.float32)) for y_i in y_list], dim=0)
    # put on GPU or device of choice
    x, y = x.to(device), y.to(device)
    return x, y 

def get_batch_test(dataset, config):
    # set parameters
    context_length = config["model"]["context_length"]
    batch_size = config["batch_size"]
    device = config["device"]
    n = len(dataset)

    # find all starting positions, skip last samples to avoid padding
    num_samples = math.floor(n / config["model"]["context_length"])
    start_positions = [i*config["model"]["context_length"] for i in range(num_samples)]
    batch_starts=[]
    for start_position in start_positions:
        batch_starts
        if len(batch_starts) == config["batch_size"]:
            batch_starts = torch.tensor(batch_starts)
            x, y = make_batch(dataset, batch_starts, context_length, config["device"])
            batch_starts = []
            yield x, y
        batch_starts.append(start_position)

    if batch_starts:
        batch_starts = torch.tensor(batch_starts)
        yield make_batch(dataset, batch_starts, context_length, config["device"])

def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int,
        device: str, valid_ids: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # set parameters
    max_sample_position = len(dataset)-context_length
    # sample starting positions
    batch_starts = torch.randint(
            low=0, high=max_sample_position, size=(batch_size,)
            ) if valid_ids is None else valid_ids 
    # make batch, keep everything in torch instead of numpy
    x_list = [dataset[batch_start:batch_start+context_length] for batch_start in batch_starts] 
    y_list = [dataset[batch_start+1:batch_start+context_length+1] for batch_start in batch_starts] 
    x = torch.stack([torch.from_numpy(np.copy(x_i).astype(np.float32)) for x_i in x_list], dim=0)
    y = torch.stack([torch.from_numpy(np.copy(y_i).astype(np.float32)) for y_i in y_list], dim=0)
    # put on GPU or device of choice
    x, y = x.to(device), y.to(device)
    return x, y 
