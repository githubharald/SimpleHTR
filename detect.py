import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from path import Path
from word_detector import detect, prepare_img, sort_multiline


def get_img_files(data_dir: Path) -> list[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res


def main():
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)

    #print("Current Directory:", current_directory)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(current_directory)/'data'/'page')
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=1080)
    parsed = parser.parse_args()

    for fn_img in get_img_files(parsed.data):
        print(f'Processing file {fn_img}')

        # load image and process it
        img = prepare_img(cv2.imread(str(fn_img)), parsed.img_height)
        detections = detect(img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma,
                            theta=parsed.theta,
                            min_area=parsed.min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)

        output_dir = './output'
        os.makedirs(output_dir, exist_ok=True)

        # save each word as a separate image
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys)
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
                output_filename = f'{os.path.splitext(fn_img.name)[0]}_{line_idx}_{word_idx}.png'
                output_path = os.path.join(output_dir, output_filename)
                separated_image = img[det.bbox.y:det.bbox.y + det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]
                separated_image = np.array(separated_image)
                cv2.imwrite(output_path, separated_image)

        # save the image with bounding boxes
       # output_filename = f'{os.path.splitext(fn_img.name)[0]}_bounding_boxes.png'
       # output_path = os.path.join(output_dir, output_filename)
       # plt.imshow(img, cmap='gray')
       # plt.savefig(output_path)
        plt.clf()


if __name__ == '__main__':
    main()