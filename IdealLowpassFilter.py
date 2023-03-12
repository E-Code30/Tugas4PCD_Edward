import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def ideal_lowpass_filter(image, radius):
    f = fftpack.fft2(image)

    fshift = fftpack.fftshift(f)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    mask = x * x + y * y <= radius * radius

    fshift[mask] = 0

    f_ishift = fftpack.ifftshift(fshift)

    image_filtered = fftpack.ifft2(f_ishift)

    image_filtered = np.real(image_filtered)

    return image_filtered

image = plt.imread("edward1.jpg")
gray_image = np.mean(image, axis=2)
filtered_image = ideal_lowpass_filter(gray_image, radius=30)
plt.imshow(filtered_image, cmap="gray")
plt.show()
