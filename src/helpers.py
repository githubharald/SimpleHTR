"""
Helper functions for handwriting recognition project
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_local, threshold_yen

SMALL_HEIGHT = 800


def preprocessor(imgPath, imgSize, binary=True):
    """ Pre-processing image for predicting """
    img = cv2.imread(imgPath)
    # Binary
    if binary:
        brightness = 0
        contrast = 50
        img = np.int16(img)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = threshold_local(img, 11, offset=10, method="gaussian")
        img = (img > T).astype("uint8") * 255

        # Increase line width
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)

    # Scale according to f (result at least 1 and at most wt or ht)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # Transpose for TF
    img = cv2.transpose(target)

    # Normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    return img


def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]
