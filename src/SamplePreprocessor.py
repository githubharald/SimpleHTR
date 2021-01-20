import random

import cv2
import numpy as np


def preprocess(img, imgSize, dataAugmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    img = img.astype(np.float)

    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        if random.random() < 0.25:
            rand_odd = lambda: random.randint(1, 3) * 2 + 1
            img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
        if random.random() < 0.25:
            img = cv2.dilate(img,np.ones((3,3)))
        if random.random() < 0.25:
            img = cv2.erode(img,np.ones((3,3)))
        if random.random() < 0.5:
            img = img * (0.25 + random.random() * 0.75)
        if random.random() < 0.25:
            img = np.clip(img + (np.random.random(img.shape)-0.5) * random.randint(1, 50), 0, 255)
        if random.random() < 0.1:
            img = 255 - img

        stretch = random.random() - 0.5  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 127.5
    
    r_freedom = target.shape[0] - img.shape[0]
    c_freedom = target.shape[1] - img.shape[1]
    
    if dataAugmentation:
        r_off, c_off = random.randint(0, r_freedom), random.randint(0, c_freedom)
    else:
        r_off, c_off = r_freedom // 2, c_freedom // 2

    target[r_off:img.shape[0]+r_off, c_off:img.shape[1]+c_off] = img

    # transpose for TF
    img = cv2.transpose(target)

    # convert to range [-1, 1]
    img = img / 255 - 0.5
    return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = cv2.imread('../data/test.png', cv2.IMREAD_GRAYSCALE)
    img_aug = preprocess(img, (128, 32), True)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(cv2.transpose(img_aug))
    plt.show()