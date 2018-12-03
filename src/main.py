import argparse
import os
import sys

import cv2
import editdistance
import numpy as np
import tensorflow as tf

from DataLoader import Batch, DataLoader, FilePaths
from helpers import preprocessor, wer
from Model import DecoderType, Model

# # Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train(model, loader):
    """ Train the neural network """
    epoch = 0  # Number of training epochs since start
    bestCharErrorRate = float('inf')  # Best valdiation character error rate
    noImprovementSince = 0  # Number of epochs no improvement of character error rate occured
    earlyStopping = 8  # Stop training after this number of epochs without improvement
    batchNum = 0

    totalEpoch = len(loader.trainSamples)//loader.numTrainSamplesPerEpoch

    while True:
        epoch += 1
        print('Epoch:', epoch, '/', totalEpoch)

        # Train
        print('Train neural network')
        loader.trainSet()
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # Validate
        charErrorRate, addressAccuracy, wordErrorRate = validate(model, loader)
        cer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
        # Tensorboard: Add cer_summary to writer
        model.writer.add_summary(cer_summary, epoch)
        address_summary = tf.Summary(value=[tf.Summary.Value(
            tag='addressAccuracy', simple_value=addressAccuracy)])  # Tensorboard: Track addressAccuracy
        # Tensorboard: Add address_summary to writer
        model.writer.add_summary(address_summary, epoch)
        wer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='wordErrorRate', simple_value=wordErrorRate)])  # Tensorboard: Track wordErrorRate
        # Tensorboard: Add wer_summary to writer
        model.writer.add_summary(wer_summary, epoch)

        # If best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # Stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' %
                  earlyStopping)
            break


def validate(model, loader):
    """ Validate neural network """
    print('Validate neural network')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0

    totalCER = []
    totalWER = []
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])

            currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
            totalCER.append(currCER)

            currWER = wer(recognized[i].split(), batch.gtTexts[i].split())
            totalWER.append(currWER)

            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' +
                  batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # Print validation result
    charErrorRate = sum(totalCER)/len(totalCER)
    addressAccuracy = numWordOK / numWordTotal
    wordErrorRate = sum(totalWER)/len(totalWER)
    print('Character error rate: %f%%. Address accuracy: %f%%. Word error rate: %f%%' %
          (charErrorRate*100.0, addressAccuracy*100.0, *100.0))
    return charErrorRate, addressAccuracy, wordErrorRate


def infer(model, fnImg):
    """ Recognize text in image provided by file path """
    img = preprocessor(fnImg, model.imgSize, binary=True)
    # Fill all batch elements with same input image
    batch = Batch(None, [img] * Model.batchSize)
    recognized = model.inferBatch(batch)  # recognize text
    # All batch elements hold same result
    print('Recognized:', '"' + recognized[0] + '"')


def main():
    """ Main function """
    # Opptional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", help="train the neural network", action="store_true")
    parser.add_argument(
        "--validate", help="validate the neural network", action="store_true")
    parser.add_argument(
        "--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # Train or validate on Cinnamon dataset
    if args.train or args.validate:
        # Load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize,
                            Model.imgSize, Model.maxTextLen)

        # Execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    # Infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(),
                      decoderType, mustRestore=False)
        infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
    main()
