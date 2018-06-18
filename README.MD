# Handwritten Text Recognition with TensorFlow

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.
This Neural Network (NN) model is trained to recognize segmented words as shown in the illustration below.
As the images of segmented words are smaller than images of complete text-lines, the NN can be kept small and therefore training on the CPU is feasible.
I will give some hints in how to extend the model in case you need larger input-images or want a better recognition accuracy.

![img](./doc/htr.png)


## Run demo

Go to the `model/` directory and unzip the file `model.zip` (pre-trained on the IAM dataset).
Afterwards, go to the `src/` directory and run ```python main.py```.
The input image and the expected output is shown below.
Tested with TF 1.3 on Ubuntu 16.04.

![img](./data/test.png)

```
> python main.py
Init with stored values from ../model/snapshot-1
Recognized: "little"
```


## Train model 

### IAM dataset

The data-loader expects the IAM dataset (or any other dataset that is compatible with it) in the `data/` directory.
Follow these instructions to get the dataset:

1. Register for free at: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
2. Download `words.tgz`
3. Download `words.txt`
4. Put `words.txt` into the `data/` directory
5. Create the directory `data/words/`
6. Put the content (directories `a01`, `a02`, ...) of `words.tgz` into `data/words/`
7. Go to `data/` and run `python checkDirs.py` for a rough check if everything is ok

If you want to train the model from scratch, delete the files contained in the `model/` directory.
Otherwise, the parameters are loaded from the last model-snapshot before training begins.
Then, go to the `src/` directory and execute `python main.py train`.
After each epoch of training, validation is done on a validation set (the dataset is split into 95% of the samples used for training and 5% for validation as defined in the class `DataLoader`).
The expected output is shown below.

```
> python main.py train
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
Correctly recognized words: 71.0 %
```

### Other datasets

Either you convert your dataset to the IAM format (look at `words.txt` and the corresponding directory structure) or you change the class `DataLoader` according to your dataset format.


## Information about model

### Overview

The model is a stripped-down version of the HTR system I used for my thesis.
What remains is what I think is the bare minimum to recognize text with an acceptable accuracy.
The implementation only depends on numpy, cv2 and tensorflow imports.
It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.
The illustration below gives an overview of the NN (green: operations, pink: data) and here follows a short description:

* The input image is a gray-value image and has a size of 128x32
* 5 CNN layers map the input image to a feature sequence of size 32x256
* 2 LSTM layers with 256 hidden units propagate information through the sequence and map the sequence to a matrix of size 32x80. Each matrix-element represents a score for one of the 80 characters at one of the 32 time-steps
* The CTC layer either calculates the loss value given the matrix and the ground-truth text (when training), or it decodes the matrix to the final text with best path decoding (when inferring)
* Batch size is set to 50

![img](./doc/nn_overview.png)


### Improve accuracy

Around 70% of the words from IAM are correctly recognized by the NN.
If you need a better accuracy, here are some ideas on how to improve it:

* Data augmentation: increase dataset-size by applying random transformations to the input images
* Remove cursive writing style in the input images (see [DeslantImg](https://github.com/githubharald/DeslantImg))
* Increase input size (if input of NN is large enough, complete text-lines can be used)
* Add more CNN layers
* Replace LSTM by MD-LSTM
* Decoder: either use vanilla beam search decoding (included with TF) or use word beam search decoding (see [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)) to constrain the output to dictionary words
* Text correction: if the recognized word is not contained in a dictionary, search for the most similar one




