import pickle
import random
from collections import namedtuple

import cv2
import lmdb
import numpy as np
from path import Path

from preprocess import preprocess

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'gt_texts, imgs')


class DataLoaderIAM:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, data_dir, batch_size, img_size, max_text_len, fast=True):
        """Loader for dataset."""

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.img_size = img_size
        self.samples = []

        f = open(data_dir / 'gt/words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            file_name_split = line_split[0].split('-')
            file_name = data_dir / 'img' / file_name_split[0] / f'{file_name_split[0]}-{file_name_split[1]}' / \
                        line_split[0] + '.png'

            if line_split[0] in bad_samples_reference:
                print('Ignoring known broken image:', file_name)
                continue

            # GT text are columns starting at 9
            gt_text = self.truncate_label(' '.join(line_split[8:]), max_text_len)
            chars = chars.union(set(list(gt_text)))

            # put sample into list
            self.samples.append(Sample(gt_text, file_name))

        # split into training and validation set: 95% - 5%
        split_idx = int(0.95 * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    @staticmethod
    def truncate_label(text, max_text_len):
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def train_set(self):
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self):
        "switch to validation set"
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self):
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self):
        "iterator"
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def get_next(self):
        "iterator"
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        imgs = []
        for i in batch_range:
            if self.fast:
                with self.env.begin() as txn:
                    basename = Path(self.samples[i].file_path).basename()
                    data = txn.get(basename.encode("ascii"))
                    img = pickle.loads(data)
            else:
                img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

            imgs.append(preprocess(img, self.img_size, self.data_augmentation))

        self.curr_idx += self.batch_size
        return Batch(gt_texts, imgs)
