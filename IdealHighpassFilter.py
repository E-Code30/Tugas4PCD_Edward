import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def ideal_highpass_filter(image, cutoff):
    f = fftpack.fft2(image)

    fshift = fftpack.fftshift(f)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    filter = np.zeros((rows, cols))
    filter[x * x + y * y >= cutoff ** 2] = 1

    fshift_filtered = fshift * filter

    f_ishift = fftpack.ifftshift(fshift_filtered)

    image_filtered = fftpack.ifft2(f_ishift)

    image_filtered = np.real(image_filtered)

    return image_filtered

image = plt.imread("edward3.jpg")
gray_image = np.mean(image, axis=2)
filtered_image = ideal_highpass_filter(gray_image, cutoff=30)
plt.imshow(filtered_image, cmap="gray")
plt.show()
