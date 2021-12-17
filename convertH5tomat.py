import h5py
import os
import scipy.io as sio
from os.path import splitext
from tqdm import tqdm
import argparse

def convert(h5path, matpath):
    h5files = os.listdir(h5path)
    os.makedirs(matpath, exist_ok=True)
    for h5f in tqdm(h5files):
        fname = splitext(h5f)[0]

        with h5py.File(os.path.join(h5path, h5f), 'r') as f:
            slices = f['data'].shape[2]
            for slice in range(slices):
                img = f['data'][..., slice]
                matfile = os.path.join(matpath, fname + '-{:03d}.mat'.format(slice))
                sio.savemat(matfile, {'img':img})

def main(root):
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    test_dir = os.path.join(root, 'test')

    matpath = root.replace('/h5/', '/mat/')
    os.makedirs(matpath, exist_ok=True)

    convert(train_dir, os.path.join(matpath, 'train'))
    convert(val_dir, os.path.join(matpath, 'val'))
    convert(test_dir, os.path.join(matpath, 'test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert h5 to mat")
    parser.add_argument(
        "--data_dir", default="", help="choose a experiment to do")
    args = parser.parse_args()
    main(args.data_dir)