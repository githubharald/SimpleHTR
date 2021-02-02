import pickle
import random

import cv2
import lmdb
import numpy as np
from path import Path

from SamplePreprocessor import preprocess


class Sample:
    "sample from the dataset"

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoaderIAM:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, data_dir, batchSize, imgSize, maxTextLen, fast=True):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open(data_dir / 'gt/words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = data_dir / 'img' / fileNameSplit[0] / f'{fileNameSplit[0]}-{fileNameSplit[1]}' / lineSplit[0] + '.png'

            if lineSplit[0] in bad_samples_reference:
                print('Ignoring known broken image:', fileName)
                continue

            # GT text are columns starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
            chars = chars.union(set(list(gtText)))

            # put sample into list
            self.samples.append(Sample(gtText, fileName))

        # split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def trainSet(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples
        self.currSet = 'train'

    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples
        self.currSet = 'val'

    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        if self.currSet == 'train':
            numBatches = int(np.floor(len(self.samples) / self.batchSize))  # train set: only full-sized batches
        else:
            numBatches = int(np.ceil(len(self.samples) / self.batchSize))  # val set: allow last batch to be smaller
        currBatch = self.currIdx // self.batchSize + 1
        return currBatch, numBatches

    def hasNext(self):
        "iterator"
        if self.currSet == 'train':
            return self.currIdx + self.batchSize <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.currIdx < len(self.samples)  # val set: allow last batch to be smaller

    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, min(self.currIdx + self.batchSize, len(self.samples)))
        gtTexts = [self.samples[i].gtText for i in batchRange]

        imgs = []
        for i in batchRange:
            if self.fast:
                with self.env.begin() as txn:
                    basename = Path(self.samples[i].filePath).basename()
                    data = txn.get(basename.encode("ascii"))
                    img = pickle.loads(data)
            else:
                img = cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE)

            imgs.append(preprocess(img, self.imgSize, self.dataAugmentation))

        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)


if __name__ == '__main__':
    dl = DataLoaderIAM('../data/', 50, (128, 32), 32)
    dl.getNext()
