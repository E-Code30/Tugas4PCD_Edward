import numpy as np
import cv2

def selective_filtering(image, kernel_size, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    abs_laplacian = np.abs(laplacian)
    abs_laplacian = np.clip(abs_laplacian, 0, 255)

    laplacian_mask = np.zeros(gray_image.shape, np.uint8)
    laplacian_mask[abs_laplacian > threshold] = 255

    median = cv2.medianBlur(laplacian_mask, kernel_size)

    filtered_image = np.zeros(image.shape, np.uint8)
    for i in range(3):
        filtered_image[:, :, i] = cv2.bitwise_and(image[:, :, i], median)

    return filtered_image

image = cv2.imread("edward3.jpg")
filtered_image = selective_filtering(image, kernel_size=5, threshold=30)
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
