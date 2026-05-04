"""
Substra opener — loads MNIST partitions saved as images.npy / labels.npy.

Each data sample is a folder containing two files:
  - images.npy : float32 array of shape (N, 1, 28, 28)
  - labels.npy : int64 array of shape (N,)

`get_data` concatenates across all folders for the same data manager and returns
a dict {"images": ..., "labels": ...}, which is what `TorchDataset` in
`substra_runner.py` expects under the `data_from_opener` argument.
"""

from pathlib import Path

import numpy as np
import substratools as tools


class MnistOpener(tools.Opener):
    def get_data(self, folders):
        images_list = []
        labels_list = []

        for folder in folders:
            f = Path(folder)
            images_list.append(np.load(f / "images.npy"))
            labels_list.append(np.load(f / "labels.npy"))

        return {
            "images": np.concatenate(images_list, axis=0),
            "labels": np.concatenate(labels_list, axis=0),
        }

    def fake_data(self, n_samples):
        return {
            "images": np.zeros((n_samples, 1, 28, 28), dtype=np.float32),
            "labels": np.zeros((n_samples,), dtype=np.int64),
        }
