import random

import cv2
import numpy as np

# TODO: change to class
# TODO: do multi-word simulation in here!


def preprocess(img, img_size, dynamic_width=False, data_augmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros(img_size[::-1])

    # data augmentation
    img = img.astype(np.float)
    if data_augmentation:
        # photometric data augmentation
        if random.random() < 0.25:
            rand_odd = lambda: random.randint(1, 3) * 2 + 1
            img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
        if random.random() < 0.25:
            img = cv2.dilate(img, np.ones((3, 3)))
        if random.random() < 0.25:
            img = cv2.erode(img, np.ones((3, 3)))
        if random.random() < 0.5:
            img = img * (0.25 + random.random() * 0.75)
        if random.random() < 0.25:
            img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 50), 0, 255)
        if random.random() < 0.1:
            img = 255 - img

        # geometric data augmentation
        wt, ht = img_size
        h, w = img.shape
        f = min(wt / w, ht / h)
        fx = f * np.random.uniform(0.75, 1.1)
        fy = f * np.random.uniform(0.75, 1.1)

        # random position around center
        txc = (wt - w * fx) / 2
        tyc = (ht - h * fy) / 2
        freedom_x = max((wt - fx * w) / 2, 0)
        freedom_y = max((ht - fy * h) / 2, 0)
        tx = txc + np.random.uniform(-freedom_x, freedom_x)
        ty = tyc + np.random.uniform(-freedom_y, freedom_y)

        # map image into target image
        M = np.float32([[fx, 0, tx], [0, fy, ty]])
        target = np.ones(img_size[::-1]) * 255 / 2
        img = cv2.warpAffine(img, M, dsize=img_size, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    # no data augmentation
    else:
        if dynamic_width:
            ht = img_size[1]
            h, w = img.shape
            f = ht / h
            wt = int(f * w)
            wt = wt + (4 - wt) % 4
            tx = (wt - w * f) / 2
            ty = 0
        else:
            wt, ht = img_size
            h, w = img.shape
            f = min(wt / w, ht / h)
            tx = (wt - w * f) / 2
            ty = (ht - h * f) / 2

        # map image into target image
        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones([ht, wt]) * 255 / 2
        img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    # transpose for TF
    img = cv2.transpose(img)

    # convert to range [-1, 1]
    img = img / 255 - 0.5
    return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('../data/test.png', cv2.IMREAD_GRAYSCALE)
    img_aug = preprocess(img, (128, 32), data_augmentation=False, dynamic_width=True)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(cv2.transpose(img_aug), cmap='gray')
    plt.show()
