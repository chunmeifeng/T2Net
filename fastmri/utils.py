"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import h5py
import os
import torch

def save_reconstructions(reconstructions, out_dir):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input
            filenames to corresponding reconstructions (of shape num_slices x
            height x width).
        out_dir (pathlib.Path): Path to the output directory where the
            reconstructions should be saved.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    print(out_dir)
    for fname, recons in reconstructions.items():
        with h5py.File(str(out_dir) + '/' + str(fname) + '.hdf5', "w") as f:
            print(fname)
            if isinstance(recons, list):
                recons = [r[1][None, ...] for r in recons]
                recons = torch.cat(recons, dim=0)
            f.create_dataset("reconstruction", data=recons)

