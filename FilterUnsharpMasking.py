import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def unsharp_masking_filter(image, kernel_size, sigma):
    kernel = np.outer(signal.gaussian(kernel_size, sigma), signal.gaussian(kernel_size, sigma))

    kernel = kernel / kernel.sum()

    highpass = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    image_highpass = signal.convolve2d(image, highpass, mode="same")

    image_blur = signal.convolve2d(image_highpass, kernel, mode="same")

    image_filtered = image + (image - image_blur)

    image_filtered = np.clip(image_filtered, 0, 255)

    return image_filtered

image = plt.imread("edward1.jpg")
gray_image = np.mean(image, axis=2)
filtered_image = unsharp_masking_filter(gray_image, kernel_size=7, sigma=2)
plt.imshow(filtered_image, cmap="gray")
plt.show()
