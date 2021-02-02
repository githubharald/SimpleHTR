# Handwritten Text Recognition with TensorFlow

* **Update 2021: more robust model, faster dataloader, Python3 only**
* **Update 2020: code is compatible with TF2**


Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.
This Neural Network (NN) model recognizes the text contained in the images of segmented words as shown in the illustration below.
3/4 of the words from the validation-set are correctly recognized and the character error rate is around 11%.

![htr](./doc/htr.png)


## Run demo
[Download the model](https://www.dropbox.com/s/lod3gabgtuj0zzn/model.zip?dl=1) trained on the IAM dataset.
Put the contents of the downloaded file `model.zip` into the `model` directory of the repository.
Afterwards, go to the `src` directory and run `python main.py`.
The input image and the expected output is shown below.

![test](./data/test.png)

```
> python main.py
Init with stored values from ../model/snapshot-76
Recognized: "Hello"
Probability: 0.8462573289871216
```

Tested with:

* Python 2  (commit <= 97c2512) and Python 3
* TF 1.3, 1.10 and 1.12 (commit <= 97c2512)
* TF 1.14, 1.15, 2.3.1, 2.4 (commit >= ec00c1a)
* Ubuntu 16.04, 18.04, 20.04 and Windows 7, 10


## Command line arguments

* `--train`: train the NN on 95% of the dataset samples and validate on the remaining 5%
* `--validate`: validate the trained NN
* `--beamsearch`: use vanilla beam search decoding (better, but slower) instead of best path decoding
* `--wordbeamsearch`: use word beam search decoding (only outputs words contained in a dictionary) instead of best path decoding. This is a custom TF operation and must be compiled from source, more information see corresponding section below. It should **not** be used when training the NN
* `--dump`: dumps the output of the NN to CSV file(s) saved in the `dump` folder. Can be used as input for the [CTCDecoder](https://github.com/githubharald/CTCDecoder)
* `--batch_size`: batch size
* `--fast`: use LMDB to load images (faster than loading image files from disk)
* `--data_dir`: directory containing IAM dataset (with subdirectories `img` and `gt`)

If neither `--train` nor `--validate` is specified, the NN infers the text from the test image (`data/test.png`).



## Integrate word beam search decoding

It is possible to use the word beam search decoder \[4\] instead of the two decoders shipped with TF.
Words are constrained to those contained in a dictionary, but arbitrary non-word character strings (numbers, punctuation marks) can still be recognized.
The following illustration shows a sample for which word beam search is able to recognize the correct text, while the other decoders fail.

![decoder_comparison](./doc/decoder_comparison.png)

Follow these instructions to integrate word beam search decoding:

1. Clone repository [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)
2. Compile custom TF operation (follow instructions given in README)
3. Copy binary `TFWordBeamSearch.so` from the CTCWordBeamSearch repository to the `src` directory of the SimpleHTR repository

Word beam search can now be enabled by setting the corresponding command line argument.
The dictionary is created (in training and validation mode) by using all words contained in the IAM dataset (i.e. also including words from validation set) and is saved into the file `data/corpus.txt`.
Further, the (manually created) list of word-characters can be found in the file `model/wordCharList.txt`.
Beam width is set to 50 to conform with the beam width of vanilla beam search decoding.


## Train model 

### IAM dataset

Follow these instructions to get the IAM dataset \[5\]:

* Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* Download `words/words.tgz`
* Download `ascii/words.txt`
* Create a directory for the dataset on your disk, and create two subdirectories: `img` and `gt`
* Put `words.txt` into the `gt` directory
* Put the content (directories `a01`, `a02`, ...) of `words.tgz` into the `img` directory

### Start the training

* Delete files from `model` directory if you want to train from scratch
* Go to the `src` directory and execute `python main.py --train --data_dir path/to/IAM`
* Training stops after a fixed number of epochs without improvement

### Fast image loading
Loading and decoding the png image files from the disk is the bottleneck even when using only a small GPU.
The database LMDB is used to speed up image loading:
* Go to the `src` directory and run `createLMDB.py --data_dir path/to/IAM` with the IAM data directory specified
* A subfolder `lmdb` is created in the IAM data directory containing the LMDB files
* When training the model, add the command line option `--fast`

Using the `--fast` option and a GTX 1050 Ti training takes around 3h with a batch size of 500.

### Other datasets

Either convert your dataset to the IAM format (look at `words.txt` and the corresponding directory structure) or change the class `DataLoaderIAM` according to your dataset format.
More information can be found in [this article](https://medium.com/@harald_scheidl/27648fb18519).


## Information about model

### Overview

The model \[1\] is a stripped-down version of the HTR system I implemented for my thesis \[2\]\[3\].
What remains is what I think is the bare minimum to recognize text with an acceptable accuracy.
It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.
The illustration below gives an overview of the NN (green: operations, pink: data flowing through NN) and here follows a short description:

* The input image is a gray-value image and has a size of 128x32
* 5 CNN layers map the input image to a feature sequence of size 32x256
* 2 LSTM layers with 256 units propagate information through the sequence and map the sequence to a matrix of size 32x80. Each matrix-element represents a score for one of the 80 characters at one of the 32 time-steps
* The CTC layer either calculates the loss value given the matrix and the ground-truth text (when training), or it decodes the matrix to the final text with best path decoding or beam search decoding (when inferring)


![nn_overview](./doc/nn_overview.png)


### Analyze model

Run `python analyze.py` with the following arguments to analyze the image file `data/analyze.png` with the ground-truth text "are":

* `--relevance`: compute the pixel relevance for the correct prediction
* `--invariance`: check if the model is invariant to horizontal translations of the text
* No argument provided: show the results

Results are shown in the plots below.
For more information see [this article](https://towardsdatascience.com/6c04864b8a98).

![analyze](./doc/analyze.png)


## FAQ

1. I get the error message "Exception: No saved model found in: ... ": unzip the file `model.zip`. All files contained must be placed directly into the `model` directory and **not** in some subdirectory created by the unzip-program.
2. I get the error message "... TFWordBeamSearch.so: cannot open shared object file: No such file or directory": if you want to use word beam search decoding, you have to compile the custom TF operation from source.
3. I get the error message "... ModuleNotFoundError: No module named 'editdistance'": you have to install the mentioned module by executing `pip install editdistance`.
4. Where can I find the file `words.txt` of the IAM dataset: it is located in the subfolder `ascii` of the IAM website.
5. I want to recognize text of line (or sentence) images: this is not possible with the provided model. The size of the input image is too small. For more information read [this article](https://medium.com/@harald_scheidl/27648fb18519) or have a look at the [lamhoangtung/LineHTR](https://github.com/lamhoangtung/LineHTR) repository.
6. I need a confidence score for the recognized text: after recognizing the text, you can calculate the loss value for the NN output and the recognized text. The loss simply is the negative logarithm of the score. See [this article](https://medium.com/@harald_scheidl/27648fb18519).
7. I use a custom image of handwritten text, but the NN outputs a wrong result: the NN is trained on the IAM dataset. The NN not only learns to recognize text, but it also learns properties of the dataset-images. Some obvious properties of the IAM dataset are: text is tightly cropped, contrast is very high, most of the characters are lower-case. Either you preprocess your image to look like an IAM image, or you train the NN on your own dataset. See [this article](https://medium.com/@harald_scheidl/27648fb18519).
8. I get an error when running the script more than once from an interactive Python session: do **not** call function `main()` in file `main.py` from an interactive session, as the TF computation graph is created multiple times when calling `main()` multiple times. Run the script by executing `python main.py` instead.
9. How to get support for this repository: I do not provide any support for this repository (also not via mail).


## References

\[1\] [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/2326a3487cd5)

\[2\] [Scheidl - Handwritten Text Recognition in Historical Documents](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742)

\[3\] [Shi - An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf)

\[4\] [Scheidl - Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578)

\[5\] [Marti - The IAM-database: an English sentence database for offline handwriting recognition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
