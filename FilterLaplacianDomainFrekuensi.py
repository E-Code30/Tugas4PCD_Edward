import numpy as np
import cv2

def laplacian_frequency_filter(image, cutoff_freq):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fft = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = gray_image.shape
    x, y = np.meshgrid(np.linspace(-cols/2, cols/2-1, cols), np.linspace(-rows/2, rows/2-1, rows))
    lp_mask = 1 - np.exp(-((x**2 + y**2) / (2 * cutoff_freq**2)))

    filtered_fft_shift = fft_shift * lp_mask

    filtered_fft = np.fft.ifftshift(filtered_fft_shift)
    filtered_image = np.fft.ifft2(filtered_fft)

    filtered_image = np.abs(filtered_image)
    filtered_image = np.clip(filtered_image, 0, 255)

    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image

image = cv2.imread("edward2.jpg")
filtered_image = laplacian_frequency_filter(image, cutoff_freq=40)
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
