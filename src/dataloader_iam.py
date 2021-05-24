import pickle
import random
from collections import namedtuple

import cv2
import lmdb
import numpy as np
from path import Path

from preprocessor import preprocess

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'gt_texts, imgs')


class DataLoaderIAM:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, data_dir, batch_size, img_size, max_text_len, fast=True, multi_word_mode=False):
        """Loader for dataset."""

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.max_text_len=max_text_len
        self.multi_word_mode = multi_word_mode

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
            gt_text = ' '.join(line_split[8:])
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
        if multi_word_mode:
            chars.add(' ')
        self.char_list = sorted(list(chars))


    @staticmethod
    def _truncate_label(text, max_text_len):
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

    def _get_img(self, i):
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    @staticmethod
    def _simulate_multi_words(imgs, gt_texts):
        batch_size = len(imgs)

        res_imgs = []
        res_gt_texts = []

        word_sep_space = 30

        for i in range(batch_size):
            j = (i + 1) % batch_size

            img_left = imgs[i]
            img_right = imgs[j]
            h = max(img_left.shape[0], img_right.shape[0])
            w = img_left.shape[1] + img_right.shape[1] + word_sep_space

            target = np.ones([h, w], np.uint8) * 255

            target[-img_left.shape[0]:, :img_left.shape[1]] = img_left
            target[-img_right.shape[0]:, -img_right.shape[1]:] = img_right

            res_imgs.append(target)
            res_gt_texts.append(gt_texts[i] + ' ' + gt_texts[j])

        return res_imgs, res_gt_texts

    def get_next(self):
        "Iterator."
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        if self.multi_word_mode:
            imgs, gt_texts = self._simulate_multi_words(imgs, gt_texts)

        # apply data augmentation to images
        imgs = [preprocess(img, self.img_size, data_augmentation=self.data_augmentation) for img in imgs]
        gt_texts = [self._truncate_label(gt_text, self.max_text_len) for gt_text in gt_texts]

        self.curr_idx += self.batch_size
        return Batch(gt_texts, imgs)
