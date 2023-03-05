import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/outputs/vines_png/00000.png', -1)

print(image.shape, image.dtype)

print(image)

# plt.imshow(image / 2**16)
# plt.show()