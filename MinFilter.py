import numpy as np
import cv2

def min_filter(image, kernel_size):
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    filtered_image = np.zeros_like(image)

    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            kernel = padded_image[i-1:i+2, j-1:j+2].flatten()
            min_val = np.min(kernel)
            filtered_image[i-1, j-1] = min_val

    return filtered_image

input_image = cv2.imread('edward2.jpg', cv2.IMREAD_GRAYSCALE)

output_image = min_filter(input_image, 3)

cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
