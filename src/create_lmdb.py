import argparse
import pickle

import cv2
import lmdb
from path import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()

# 2GB is enough for IAM dataset
assert not (args.data_dir / 'lmdb').exists()
env = lmdb.open(str(args.data_dir / 'lmdb'), map_size=1024 * 1024 * 1024 * 2)

# go over all png files
fn_imgs = list((args.data_dir / 'img').walkfiles('*.png'))

# and put the imgs into lmdb as pickled grayscale imgs
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(i, len(fn_imgs))
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        basename = fn_img.basename()
        txn.put(basename.encode("ascii"), pickle.dumps(img))

env.close()
