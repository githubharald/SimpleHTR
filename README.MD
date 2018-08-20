# Handwritten Text Recognition with TensorFlow

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.
This Neural Network (NN) model recognizes the text contained in the images of segmented words as shown in the illustration below.
As these word-images are smaller than images of complete text-lines, the NN can be kept small and training on the CPU is feasible.
2/3 of the words from the validation-set are correctly recognized and the character error rate is around 13%.
I will give some hints how to extend the model in case you need larger input-images (e.g. to recognize text-lines) or want better recognition accuracy.

![img](./doc/htr.png)


## Run demo

Go to the `model/` directory and unzip the file `model.zip` (pre-trained on the IAM dataset).
Afterwards, go to the `src/` directory and run ```python main.py```.
The input image and the expected output is shown below.
Tested with TF 1.3 on Ubuntu 16.04.

![img](./data/test.png)

```
> python main.py
Validation character error rate of saved model: 13.956289%
Init with stored values from ../model/snapshot-32
Recognized: "little"
```


## Command line arguments

* `--train`: train the NN, details see below.
* `--validate`: validate the NN, details see below.
* `--beamsearch`: use beam search decoding (better, but slower) instead of best path decoding.

If neither `--train` nor `--validate` is specified, the NN infers the text from the test image (`data/test.png`).
Two examples: if you want to infer using beam search, execute `python main.py --beamsearch`, while you have to execute `python main.py --train --beamsearch` if you want to train the NN and do the validation using beam search.


## Train model 

### IAM dataset

The data-loader expects the IAM dataset (or any other dataset that is compatible with it) in the `data/` directory.
Follow these instructions to get the dataset:

1. Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
2. Download `words/words.tgz`.
3. Download `ascii/words.txt`.
4. Put `words.txt` into the `data/` directory.
5. Create the directory `data/words/`.
6. Put the content (directories `a01`, `a02`, ...) of `words.tgz` into `data/words/`.
7. Go to `data/` and run `python checkDirs.py` for a rough check if everything is ok.

If you want to train the model from scratch, delete the files contained in the `model/` directory.
Otherwise, the parameters are loaded from the last model-snapshot before training begins.
Then, go to the `src/` directory and execute `python main.py --train`.
After each epoch of training, validation is done on a validation set (the dataset is split into 95% of the samples used for training and 5% for validation as defined in the class `DataLoader`).
If you only want to do validation given a trained NN, execute `python main.py --validate`.
Training on the CPU takes 18 hours on my system (VM, Ubuntu 16.04, 8GB of RAM and 4 cores running at 3.9GHz).
The expected output is shown below.

```
> python main.py --train
Init with new values
Epoch: 1
Train NN
Batch: 1 / 500 Loss: 130.354
Batch: 2 / 500 Loss: 66.6619
Batch: 3 / 500 Loss: 36.0154
Batch: 4 / 500 Loss: 24.5898
Batch: 5 / 500 Loss: 20.1845
Batch: 6 / 500 Loss: 19.2857
Batch: 7 / 500 Loss: 18.3493
...

Validate NN
Batch: 1 / 115
Ground truth -> Recognized
[OK] "," -> ","
[ERR:1] "Di" -> "D"
[OK] "," -> ","
[OK] """ -> """
[OK] "he" -> "he"
[OK] "told" -> "told"
[ERR:2] "her" -> "nor"
...
Character error rate: 13.956289%. Word accuracy: 67.721739%.
```

### Other datasets

Either convert your dataset to the IAM format (look at `words.txt` and the corresponding directory structure) or change the class `DataLoader` according to your dataset format.


## Information about model

### Overview

The model is a stripped-down version of the HTR system I implemented for my thesis.
What remains is what I think is the bare minimum to recognize text with an acceptable accuracy.
The implementation only depends on numpy, cv2 and tensorflow imports.
It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.
The illustration below gives an overview of the NN (green: operations, pink: data flowing through NN) and here follows a short description:

* The input image is a gray-value image and has a size of 128x32
* 5 CNN layers map the input image to a feature sequence of size 32x256
* 2 LSTM layers with 256 units propagate information through the sequence and map the sequence to a matrix of size 32x80. Each matrix-element represents a score for one of the 80 characters at one of the 32 time-steps
* The CTC layer either calculates the loss value given the matrix and the ground-truth text (when training), or it decodes the matrix to the final text with best path decoding or beam search decoding (when inferring)
* Batch size is set to 50

![img](./doc/nn_overview.png)


### Improve accuracy

Around 68% of the words from the IAM dataset are correctly recognized by the NN when using beam search decoding.
If you need a better accuracy, here are some ideas how to improve it:

* Data augmentation: increase dataset-size by applying further (random) transformations to the input images. At the moment, only random distortions are performed.
* Remove cursive writing style in the input images (see [DeslantImg](https://github.com/githubharald/DeslantImg)).
* Increase input size (if input of NN is large enough, complete text-lines can be used).
* Add more CNN layers.
* Replace LSTM by 2D-LSTM.
* Decoder: use token passing or word beam search decoding (see [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)) to constrain the output to dictionary words.
* Text correction: if the recognized word is not contained in a dictionary, search for the most similar one.


## FAQ

* I get the error message "Exception: No saved model found in: ... ": unzip the file `model/model.zip`. All files contained must be placed directly into the `model/` directory and **not** in some subdirectory created by the unzip-program.
* I want to recognize text for complete lines/sentences: you have to increase the size of NN, especially the width of the input image (e.g. 800px) and the number of time-steps (e.g. 100) of the output matrix.
* I need a confidence score for the recognized text: after recognizing the text, you can calculate the loss value for the NN output and the recognized text. The loss simply is the negative logarithm of the score.
* I use a custom image of handwritten text, but the NN outputs a wrong result: the NN is trained on the IAM dataset. The NN not only learns to recognize text, but it also learns properties of the dataset-images. Some obvious properties of the IAM dataset are: text is tightly cropped, contrast is very high, most of the characters are lower-case. Either you preprocess your image to look like an IAM image, or you train the NN on your own dataset.
* Where can I find the file `words.txt` of the IAM dataset: it is located in the subfolder `ascii` of the IAM website.

