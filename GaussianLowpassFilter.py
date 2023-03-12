import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def gaussian_lowpass_filter(image, sigma):
    f = fftpack.fft2(image)

    fshift = fftpack.fftshift(f)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    filter = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    fshift_filtered = fshift * filter

    f_ishift = fftpack.ifftshift(fshift_filtered)

    image_filtered = fftpack.ifft2(f_ishift)

    image_filtered = np.real(image_filtered)

    return image_filtered

image = plt.imread("edward3.jpg")
gray_image = np.mean(image, axis=2)
filtered_image = gaussian_lowpass_filter(gray_image, sigma=5)
plt.imshow(filtered_image, cmap="gray")
plt.show()
