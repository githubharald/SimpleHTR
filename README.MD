# Handwritten Text Recognition with TensorFlow

Handwritten Text Recognition (HTR) system implemented in TensorFlow (TF) and trained on the IAM offline HTR dataset.
This Neural Network (NN) implementation is the bare minimum that is needed to detect handwritten text with TF.
It is trained to recognize segmented words, therefore the model can be kept small and training on the CPU is feasible.
If you want to get a higher recognition accuracy or if you want to input larger images (e.g. images of text-lines), I will give some hints how to enhance the model.

![img](file://doc/htr.png)


## Run demo

Go to the `model/` directory and unzip the file `model.zip` (this model is pre-trained on the IAM dataset).
Afterwards, go to the `src/` directory and run ```python main.py```.

The input image and the expected output is shown below:

![img](file://data/test.png)

```
Init with stored values from ../model/snapshot-1
Recognized: "house"
```


## Train new model on IAM dataset

The data-loader expects the IAM dataset (or any other dataset that is compatible with it) in the `data/` directory.
Follow these instructions to get the dataset:

1. Register at: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
2. Download `words.tgz`
3. Download `words.txt`
4. Put `words.txt` into the `data/` directory
5. Create the directory `data/words/`
6. Put the content (directories `a01`, `a02`, ...) of `words.tgz` into `data/words/`
7. Go to `data/` and run `python checkDirs.py` for a rough check if everything is ok

If you want to initialize the model with new parameter values, delete the files contained in the `model/` directory.
Otherwise, keep them to continue training.
Go to the `src/` directory and execute `python main.py train`.
The expected output is shown below.
After each epoch of training, validation is done on a validation set (the dataset is split into 95% of the samples used for training and 5% for validation).

```
Init with stored values from ../model/snapshot-1
Epoch: 0
Train NN
Batch: 0 / 2191 Loss: 3.87954
Batch: 1 / 2191 Loss: 5.31012
Batch: 2 / 2191 Loss: 3.87662
Batch: 3 / 2191 Loss: 4.03646
...

Validate NN
Batch: 0 / 115
Ground truth -> Recognized
[OK] "," -> ","
[ERR] "Di" -> "D"
[OK] "," -> ","
[OK] """ -> """
[OK] "he" -> "he"
[OK] "told" -> "told"
[OK] "her" -> "her"
...
Correctly recognized words: 60.0 %
```

# Train new model on another dataset

Either you convert your dataset into the IAM format (look at words.txt and the corresponding directory structure) or you change the class `DataLoader` according to your dataset format.


# Overview of the model

The model is a stripped-down version of the HTR system I used for my thesis.
It only depends on numpy, cv2 and tensorflow imports.
It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.
The illustration below gives an overview of the operations and tensors of the NN, here follows a short description:

* The input image is gray-valued and has a size of 128x32.
* 5 CNN layers map the input image to a feature sequence of size 32x256.
* 2 LSTM layers propagate information through the sequence and map the sequence to a matrix of size 32x80. Each matrix-element represents a score for one of the 80 characters at one of the 32 time-steps.
* The CTC layer either calcualtes the loss value given the matrix and the ground-truth text, or it decodes the matrix to the final text with best path decoding.

![img](file://doc/nn_overview.png)


# How to enhance the model

* Increase size of input image (if input of NN is large enough, also complete text-lines can be used)
* Add more CNN layers
* Data augmentation: increase size of dataset by doing random transformations to the input images
* Remove the cursive writing style in the input images (see [DeslantImg](https://github.com/githubharald/DeslantImg))
* Decoder: either use vanilla beam search decoding (included with TF) or use word beam search decoding (see [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)) to constrain output to words from a dictionary
* Text correction: if the recognized is not contained in a dictionary, the most similar one can be taken instead




