import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import multiprocessing

import jxlbinding

from time import time, sleep
from tqdm import tqdm


def get_data(dirpath_input, dirpath_output, batch_size):
    images = []
    filepath_outputs = []
    for file_input in tqdm(sorted(os.listdir(dirpath_input))[:batch_size]):
        filepath_input = os.path.join(dirpath_input, file_input)
        image = cv2.imread(filepath_input)
        image = (image * 256).astype(np.uint16)
        images.append(image)

        filepath_output = os.path.join(dirpath_output, file_input[:-4] + ".jxl")
        filepath_outputs.append(filepath_output)

    return images, filepath_outputs


if __name__ == "__main__":

    super_batch = 20
    sub_batch = 9

    batch_size = super_batch*sub_batch

    images, filepath_outputs = get_data("./images/inputs", "./images/outputs/vines", batch_size)

    images_ = []
    filepath_outputs_ = []

    for i in range(super_batch):
        images_.append([])
        filepath_outputs_.append([])
        for j in range(sub_batch):
            images_[i].append(images[i*sub_batch+j])
            filepath_outputs_[i].append(filepath_outputs[i*sub_batch+j])

    t0 = time()
    jxlbinding.encode_images(images_, filepath_outputs_)
    t1 = time()

    total_pixels = 0
    for image in images:
        total_pixels += image.shape[0]*image.shape[1]

    print(f"time: {t1 - t0:.3f} - MP/s: {total_pixels / (t1 - t0) / 1e6:.3f}")
    # print("out", out)