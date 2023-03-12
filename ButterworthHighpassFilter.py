import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def butterworth_highpass_filter(image, cutoff, order):
    f = fftpack.fft2(image)

    fshift = fftpack.fftshift(f)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    distance = np.sqrt(x ** 2 + y ** 2)
    filter = 1 / (1 + (distance / cutoff) ** (2 * order))

    fshift_filtered = fshift * filter

    f_ishift = fftpack.ifftshift(fshift_filtered)

    image_filtered = fftpack.ifft2(f_ishift)

    image_filtered = np.real(image_filtered)

    return image_filtered

image = plt.imread("edward1.jpg")
gray_image = np.mean(image, axis=2)
filtered_image = butterworth_highpass_filter(gray_image, cutoff=30, order=2)
plt.imshow(filtered_image, cmap="gray")
plt.show()
