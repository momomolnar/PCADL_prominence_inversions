import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from s4pi.irradiance.models.model import IrradianceModel, ChoppedAlexnetBN, LinearIrradianceModel, HybridIrradianceModel, unnormalize
from s4pi.irradiance.utilities.data_loader import FITSDataset, NumpyDataset

def ipredict(model, dataset, return_images=False, batch_size=2, num_workers = None):
    """Predict irradiance for a given set of npy image stacks using a generator.

    Parameters
    ----------
    chk_path: model save point.
    dataset: pytorch dataset for streaming the input data.
    return_images: set True to return input images.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    predicted irradiance as numpy array and corresponding image if return_images==True
    """
    # use model after training or load weights and drop into the production system
    model.eval()
    # load data
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(model.device)
            pred_irradiance = model.forward_unnormalize(imgs)
            for pred, img in zip(pred_irradiance, imgs):
                if return_images:
                    yield pred.cpu(), img.cpu()
                else:
                    yield pred.cpu()

